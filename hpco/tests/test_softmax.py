import torch


def native_safe_softmax(x):
    x_max, _ = x.max(axis=1)
    x = x - x_max[:, None]
    x = x.exp()
    den = x.sum(axis=1)
    return x / den[:, None]


if __name__ == "__main__":
    # x = torch.randn(3, 4)
    x = torch.tensor([range(64)]).reshape(1, 64)
    out = native_safe_softmax(x)
    print(x, out)
