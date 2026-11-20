# EdgeInfer - On-Device AI Inference Engine in C++17

A tiny neural-network inference engine written from scratch in C++ (no ML libraries).
It runs the same model in **float32** and **int8 (quantized)** and compares accuracy,
latency and model size - the core idea behind running AI on phones and edge chips.

## What it does
- Loads a trained model (binary format) and runs inference in pure C++17
- **Post-training int8 quantization**: symmetric weight quantization + dynamic activation
  quantization, with int32 integer multiply-accumulate
- **Multi-threaded** batch inference using `std::thread`
- Built-in benchmark: accuracy, latency per sample, model size, prediction agreement
- Python is used only once, to train the model and export weights (`train.py`)

## Results (handwritten digits, 540 test samples, 1 CPU core, g++ -O3)
| Mode    | Accuracy | Latency/sample | Model size |
|---------|----------|----------------|------------|
| float32 | 97.96%   | ~21 us         | 198.5 KB   |
| int8    | 97.96%   | ~10 us         | 50.8 KB    |

int8 gives **3.9x smaller model**, about **2x faster**, with **no accuracy loss**.
(Numbers vary by machine.)

## Run on Linux
```bash
pip install numpy scikit-learn
python3 train.py                 # trains and creates model.bin + test.bin
make                             # builds with g++
./edgeinfer --rounds 300 --threads 4
```

## Concepts used
C++17, STL (vector, thread, chrono), binary file I/O, pointers and memory layout,
integer arithmetic, quantization, multithreading, Linux build with Makefile.

## Ideas to extend
- Add AVX2/NEON SIMD intrinsics for the int8 inner loop
- Support convolution layers, per-channel quantization
- Profile with `perf` and report cache behaviour
