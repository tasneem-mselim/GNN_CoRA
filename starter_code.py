import os, random
import numpy as np
import pandas as pd
import tensorflow as tf

DATA_DIR = "" 
TEST_SUBMISSION_PATH = "" 
EDGE_PATH = os.path.join("edge_index.csv")
X_PATH = os.path.join("x.csv")
YTR_PATH = os.path.join("y_train.csv")
YVA_PATH = os.path.join("y_val.csv")
OUT_PATH ="TEST_SUBMISSION_PATH"

SEED = 25
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


X = pd.read_csv(X_PATH).to_numpy(dtype=np.float32)  # (N,F)
N, F = X.shape

e = pd.read_csv(EDGE_PATH)
src = e["source"].to_numpy(dtype=np.int64)
dst = e["target"].to_numpy(dtype=np.int64)

good = (src >= 0) & (src < N) & (dst >= 0) & (dst < N)
src, dst = src[good], dst[good]
tr = pd.read_csv(YTR_PATH)
va = pd.read_csv(YVA_PATH)

tr_idx = tr["index"].to_numpy(dtype=np.int64)
tr_y   = tr["label"].to_numpy(dtype=np.int64)

va_idx = va["index"].to_numpy(dtype=np.int64)
va_y   = va["label"].to_numpy(dtype=np.int64)

# number of classes
C = int(max(tr_y.max(initial=0), va_y.max(initial=0))) + 1

# build one-hot labels for all nodes + masks
Y = np.zeros((N, C), dtype=np.float32)
train_mask = np.zeros((N,), dtype=np.float32)
val_mask   = np.zeros((N,), dtype=np.float32)

Y[tr_idx, tr_y] = 1.0
train_mask[tr_idx] = 1.0

Y[va_idx, va_y] = 1.0
val_mask[va_idx] = 1.0
row = np.concatenate([src, dst, np.arange(N, dtype=np.int64)])
col = np.concatenate([dst, src, np.arange(N, dtype=np.int64)])
val = np.ones_like(row, dtype=np.float32)

idx = row * N + col
order = np.argsort(idx)
row, col, val = row[order], col[order], val[order]
_, first = np.unique(row * N + col, return_index=True)
val_sum = np.add.reduceat(val, first)
row_u, col_u, val_u = row[first], col[first], val_sum

deg = np.zeros(N, dtype=np.float32)
np.add.at(deg, row_u, val_u)
deg_inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
norm_val = deg_inv_sqrt[row_u] * val_u * deg_inv_sqrt[col_u]

A_hat = tf.sparse.SparseTensor(
    indices=np.stack([row_u, col_u], axis=1),
    values=norm_val.astype(np.float32),
    dense_shape=(N, N),
)
A_hat = tf.sparse.reorder(A_hat)


X_in = tf.keras.Input(shape=(F,), name="X")
A_in = tf.keras.Input(shape=(None,), sparse=True, name="A_hat")

def spmm(inputs):
    h, a = inputs
    return tf.sparse.sparse_dense_matmul(a, h)

propagate = tf.keras.layers.Lambda(spmm, output_shape=lambda input_shapes: input_shapes[0], name="propagate")

h = tf.keras.layers.Dense(16, use_bias=False)(X_in)
h = propagate([h, A_in])
h = tf.keras.layers.Activation("relu")(h)
h = tf.keras.layers.Dropout(0.5, seed=SEED)(h)

h = tf.keras.layers.Dense(C, use_bias=False)(h)
h = propagate([h, A_in])
out = tf.keras.layers.Activation("softmax")(h)

model = tf.keras.Model([X_in, A_in], out)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss="categorical_crossentropy",
    weighted_metrics=["accuracy"],
)

model.fit(
    x=[X, A_hat],
    y=Y,
    sample_weight=train_mask,                  # train loss on train nodes only
    validation_data=([X, A_hat], Y, val_mask), # val metrics on val nodes only
    epochs=5,
    batch_size=N,                             
    verbose=2,
)

proba = model.predict([X, A_hat], batch_size=N, verbose=0)
pred = proba.argmax(axis=1).astype(np.int64)

test_ids = pd.read_csv(TESTID_PATH)["id"].to_numpy(dtype=np.int64)
sub = pd.DataFrame({"id": test_ids, "target": pred[test_ids]})
sub.to_csv(SUB_PATH, index=False)
print("Saved submission:", SUB_PATH)

# quick val accuracy (post-hoc)
val_acc = (pred[va_idx] == va_y).mean()
print(f"Val accuracy (post-hoc): {val_acc:.4f}")
