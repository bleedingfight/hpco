import numpy as np
import torch.nn.functional as F
import torch


def native_softmax(Q):
    max = np.max(Q, axis=1)
    Q = Q - max[:, None]
    exp_q = np.exp(Q)
    return exp_q / np.sum(exp_q, axis=1)[:, None]


def online_softmax1D(row_data):
    g_m = float("-inf")
    d = 0
    for x in row_data:
        d = d * np.exp(g_m - np.maximum(x, g_m)) + np.exp(x -
                                                          np.maximum(g_m, x))
        g_m = np.maximum(x, g_m)
    out = [np.exp(x - g_m) for x in row_data]
    return [x / d for x in out]


def online_softmax(Q):
    row, col = Q.shape
    value = []
    for r in range(row):
        value.append(online_softmax1D(Q[r, :]))
    return np.stack(value)


def native_attention(Q, K, V):
    # breakpoint()
    qk = np.matmul(Q, K.T)
    qk = qk / np.sqrt(K.shape[1])  # Scale the dot product
    qk_s = native_softmax(qk)
    o = np.matmul(qk_s, V)
    return o


def flashattention(Q_Matrix, K_Matrix, V_Matrix):
    assert Q_Matrix.shape[1] == K_Matrix.shape[1], (
        f"Q 的行应该和 K 的行相同，但是当前 Q:{Q_Matrix.shape[0]} K{K_Matrix.shape[0]}")
    M, K = Q_Matrix.shape
    N, _ = K_Matrix.shape
    L = V_Matrix.shape[1]
    out = np.zeros((M, L))

    for m in range(M):
        q_row = Q_Matrix[m, :]
        out_row = []
        g_m = float("-inf")
        d = 0
        for n in range(N):
            acc = np.dot(q_row, K_Matrix[n, :]) / np.sqrt(K_Matrix.shape[1])
            out_row.append(acc)
            d = d * np.exp(g_m - max(acc, g_m)) + np.exp(acc - max(acc, g_m))
            g_m = max(acc, g_m)
        out_row = np.exp((np.array(out_row) - g_m))
        out_row = out_row / np.sum(out_row)
        for l in range(L):
            out[m, l] = np.dot(out_row, V_Matrix[:, l])
    return out


def np_to_file(array, filename):
    with open(filename, 'w') as f:
        for e in array.flatten().tolist():
            f.write(f"{e} ")


if __name__ == "__main__":
    M = 4
    K = 5
    N = 8
    L = 4
    np.random.seed(0)
    Q = np.random.randint(0, 10, (M, K)).astype(np.float32)
    K = np.random.randint(0, 10, (N, K)).astype(np.float32)
    V = np.random.randint(0, 10, (N, L)).astype(np.float32)
    np_to_file(Q, "Q.txt")
    np_to_file(K, "K.txt")
    np_to_file(V, "V.txt")
    O = native_attention(Q, K, V)
    O1 = flashattention(Q, K, V)
    np.testing.assert_allclose(O, O1, rtol=1e-5, atol=1e-5)
    # sm_1 = native_softmax(Q)
    # sm_2 = online_softmax(Q)
    # print(O, O.shape, O1, O1.shape)
    # print(f"native softmax = {sm_1} online softmax = {sm_2}")
    output = F.scaled_dot_product_attention(torch.from_numpy(Q),
                                            torch.from_numpy(K),
                                            torch.from_numpy(V))
    np.testing.assert_allclose(O, output, rtol=1e-5, atol=1e-5)
    print(output, output.shape, output.dtype)
