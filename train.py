"""Train a small neural network on handwritten digits and export it
as plain binary files that the C++ engine can load (no Python needed at runtime)."""
import struct
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

X, y = load_digits(return_X_y=True)
X = (X / 16.0).astype(np.float32)            # pixels 0..16 -> 0..1
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

clf = MLPClassifier(hidden_layer_sizes=(256, 128), activation="relu",
                    max_iter=300, random_state=1)
clf.fit(Xtr, ytr)
print(f"Python (sklearn) test accuracy: {clf.score(Xte, yte) * 100:.2f}%")

# model.bin: int32 nLayers, then per layer: int32 in, int32 out,
#            float32 W[out][in] (row-major), float32 b[out]
with open("model.bin", "wb") as f:
    f.write(struct.pack("<i", len(clf.coefs_)))
    for W, b in zip(clf.coefs_, clf.intercepts_):
        n_in, n_out = W.shape
        f.write(struct.pack("<ii", n_in, n_out))
        f.write(np.ascontiguousarray(W.T, dtype=np.float32).tobytes())
        f.write(b.astype(np.float32).tobytes())

# test.bin: int32 n, int32 dim, float32 X[n][dim], int32 y[n]
with open("test.bin", "wb") as f:
    f.write(struct.pack("<ii", *Xte.shape))
    f.write(np.ascontiguousarray(Xte, dtype=np.float32).tobytes())
    f.write(yte.astype(np.int32).tobytes())
print("Saved model.bin and test.bin")
