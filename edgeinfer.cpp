// EdgeInfer - a tiny on-device AI inference engine in modern C++17.
// Runs the same neural network in float32 and in int8 (quantized),
// then compares accuracy, speed and model size. Linux, no dependencies.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

struct Layer {
    int in = 0, out = 0;
    std::vector<float> W, b;      // float weights, W[out][in]
    std::vector<int8_t> Wq;       // int8 weights
    float wscale = 1.0f;          // dequantization scale for weights
};

struct Model {
    std::vector<Layer> layers;

    static Model load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("cannot open " + path);
        Model m;
        int32_t n = 0;
        f.read(reinterpret_cast<char*>(&n), 4);
        m.layers.resize(n);
        for (auto& L : m.layers) {
            int32_t a, b;
            f.read(reinterpret_cast<char*>(&a), 4);
            f.read(reinterpret_cast<char*>(&b), 4);
            L.in = a; L.out = b;
            L.W.resize(size_t(a) * b);
            L.b.resize(b);
            f.read(reinterpret_cast<char*>(L.W.data()), L.W.size() * 4);
            f.read(reinterpret_cast<char*>(L.b.data()), L.b.size() * 4);
        }
        if (!f) throw std::runtime_error("model file is corrupted");
        return m;
    }

    // Symmetric int8 quantization: w ~= wq * scale, wq in [-127, 127]
    void quantize() {
        for (auto& L : layers) {
            float mx = 0.f;
            for (float w : L.W) mx = std::max(mx, std::fabs(w));
            L.wscale = mx > 0 ? mx / 127.f : 1.f;
            L.Wq.resize(L.W.size());
            for (size_t i = 0; i < L.W.size(); ++i)
                L.Wq[i] = static_cast<int8_t>(std::lround(L.W[i] / L.wscale));
        }
    }

    size_t bytesFloat() const {
        size_t s = 0;
        for (auto& L : layers) s += (L.W.size() + L.b.size()) * 4;
        return s;
    }
    size_t bytesInt8() const {   // int8 weights + float biases + 1 scale/layer
        size_t s = 0;
        for (auto& L : layers) s += L.Wq.size() + L.b.size() * 4 + 4;
        return s;
    }
};

// ---- forward passes (return predicted class) ----
static int predictFloat(const Model& m, const float* x, std::vector<float>& a,
                        std::vector<float>& b) {
    a.assign(x, x + m.layers[0].in);
    for (size_t l = 0; l < m.layers.size(); ++l) {
        const Layer& L = m.layers[l];
        b.assign(L.out, 0.f);
        for (int o = 0; o < L.out; ++o) {
            const float* w = &L.W[size_t(o) * L.in];
            float s = L.b[o];
            for (int i = 0; i < L.in; ++i) s += w[i] * a[i];
            b[o] = (l + 1 < m.layers.size()) ? std::max(0.f, s) : s;  // ReLU
        }
        a.swap(b);
    }
    return int(std::max_element(a.begin(), a.end()) - a.begin());
}

static int predictInt8(const Model& m, const float* x, std::vector<float>& a,
                       std::vector<float>& b, std::vector<int8_t>& q) {
    a.assign(x, x + m.layers[0].in);
    for (size_t l = 0; l < m.layers.size(); ++l) {
        const Layer& L = m.layers[l];
        // quantize the activations on the fly (dynamic quantization)
        float mx = 0.f;
        for (float v : a) mx = std::max(mx, std::fabs(v));
        float xscale = mx > 0 ? mx / 127.f : 1.f;
        q.resize(L.in);
        for (int i = 0; i < L.in; ++i)
            q[i] = static_cast<int8_t>(std::lround(a[i] / xscale));
        b.assign(L.out, 0.f);
        for (int o = 0; o < L.out; ++o) {
            const int8_t* w = &L.Wq[size_t(o) * L.in];
            int32_t acc = 0;                       // integer multiply-accumulate
            for (int i = 0; i < L.in; ++i) acc += int32_t(w[i]) * int32_t(q[i]);
            float s = acc * L.wscale * xscale + L.b[o];
            b[o] = (l + 1 < m.layers.size()) ? std::max(0.f, s) : s;
        }
        a.swap(b);
    }
    return int(std::max_element(a.begin(), a.end()) - a.begin());
}

struct TestSet {
    int n = 0, dim = 0;
    std::vector<float> X;
    std::vector<int32_t> y;
    static TestSet load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("cannot open " + path);
        TestSet t;
        f.read(reinterpret_cast<char*>(&t.n), 4);
        f.read(reinterpret_cast<char*>(&t.dim), 4);
        t.X.resize(size_t(t.n) * t.dim);
        t.y.resize(t.n);
        f.read(reinterpret_cast<char*>(t.X.data()), t.X.size() * 4);
        f.read(reinterpret_cast<char*>(t.y.data()), t.y.size() * 4);
        if (!f) throw std::runtime_error("test file is corrupted");
        return t;
    }
};

struct Result { int correct = 0; double ms = 0; std::vector<int> pred; };

// Run the whole test set `rounds` times, split across `threads` threads.
static Result run(const Model& m, const TestSet& t, bool int8, int rounds, int threads) {
    Result r;
    r.pred.assign(t.n, -1);
    auto t0 = std::chrono::steady_clock::now();
    for (int rd = 0; rd < rounds; ++rd) {
        auto work = [&](int from, int to) {
            std::vector<float> a, b;
            std::vector<int8_t> q;
            for (int i = from; i < to; ++i) {
                const float* x = &t.X[size_t(i) * t.dim];
                r.pred[i] = int8 ? predictInt8(m, x, a, b, q) : predictFloat(m, x, a, b);
            }
        };
        std::vector<std::thread> pool;
        int chunk = (t.n + threads - 1) / threads;
        for (int k = 0; k < threads; ++k)
            pool.emplace_back(work, k * chunk, std::min(t.n, (k + 1) * chunk));
        for (auto& th : pool) th.join();
    }
    auto t1 = std::chrono::steady_clock::now();
    r.ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    for (int i = 0; i < t.n; ++i) r.correct += (r.pred[i] == t.y[i]);
    return r;
}

int main(int argc, char** argv) {
    int rounds = 200, threads = 1;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--rounds") && i + 1 < argc) rounds = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--threads") && i + 1 < argc) threads = std::atoi(argv[++i]);
        else { std::cerr << "usage: edgeinfer [--rounds N] [--threads N]\n"; return 1; }
    }
    if (rounds < 1) rounds = 1;
    if (threads < 1) threads = 1;
    try {
        Model m = Model::load("model.bin");
        TestSet t = TestSet::load("test.bin");
        m.quantize();

        Result f = run(m, t, false, rounds, threads);
        Result q = run(m, t, true, rounds, threads);

        int same = 0;
        for (int i = 0; i < t.n; ++i) same += (f.pred[i] == q.pred[i]);
        double calls = double(rounds) * t.n;

        std::printf("\n=== EdgeInfer: float32 vs int8 on-device inference ===\n");
        std::printf("Samples: %d | Rounds: %d | Threads: %d\n\n", t.n, rounds, threads);
        std::printf("%-10s %10s %14s %12s\n", "Mode", "Accuracy", "Latency/sample", "Model size");
        std::printf("%-10s %9.2f%% %11.2f us %9.1f KB\n", "float32",
                    100.0 * f.correct / t.n, f.ms * 1000.0 / calls, m.bytesFloat() / 1024.0);
        std::printf("%-10s %9.2f%% %11.2f us %9.1f KB\n", "int8",
                    100.0 * q.correct / t.n, q.ms * 1000.0 / calls, m.bytesInt8() / 1024.0);
        std::printf("\nModel size reduced %.1fx | int8 speed vs float32: %.2fx\n",
                    double(m.bytesFloat()) / m.bytesInt8(), f.ms / q.ms);
        std::printf("int8 and float32 agree on %.2f%% of predictions\n", 100.0 * same / t.n);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n(run: python3 train.py first)\n";
        return 1;
    }
    return 0;
}
