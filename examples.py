from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from ot_pytorch import sink, sink_stabilized


def dmat(x, y):
    mmp1 = torch.stack([x] * x.size()[0])
    mmp2 = torch.stack([y] * y.size()[0]).transpose(0, 1)
    mm = torch.sum((mmp1 - mmp2) ** 2, 2).squeeze()

    return mm


def uniform_example(batch_size=100, reg=10, filename="uniform_example1", device="cpu"):
    m_list = (np.array(list(range(1, 100))) / 50.0 - 1).tolist()
    loss = []
    for theta in m_list:
        x = np.zeros((batch_size, 2))
        y = np.zeros((batch_size, 2))
        x[:, 1] = np.random.uniform(0, 1, batch_size)
        y[:, 1] = np.random.uniform(0, 1, batch_size)
        y[:, 0] = theta

        x = torch.from_numpy(x).float().to(device)
        y = torch.from_numpy(y).float().to(device)

        M = dmat(x, y)
        out = sink(M, reg=reg)
        out.backward()
        loss.append(out.detach().cpu().numpy())

    plt.plot(m_list, loss)
    plt.xlabel("Theta")
    plt.ylabel("Sinkhorn Distance")
    plt.title("Uniform Example")
    fig_name = "plots/uniform_example/" + filename + ".png"
    Path(fig_name).parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(fig_name)
    plt.show()

    df = pd.DataFrame({"theta": m_list, "sink_dist": loss})
    data_name = "data/uniform_example/" + filename + ".csv"
    Path(data_name).parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(data_name)


def uniform_stabilized_example(batch_size=100, reg=10, filename="uniform_stabilized_example1", device="cpu"):
    m_list = (np.array(list(range(1, 100))) / 50.0 - 1).tolist()
    loss = []
    for theta in m_list:
        x = np.zeros((batch_size, 2))
        y = np.zeros((batch_size, 2))
        x[:, 1] = np.random.uniform(0, 1, batch_size)
        y[:, 1] = np.random.uniform(0, 1, batch_size)
        y[:, 0] = theta

        x = torch.from_numpy(x).float().to(device)
        y = torch.from_numpy(y).float().to(device)

        M = dmat(x, y)
        out = sink_stabilized(M, reg=reg)
        out.backward()
        loss.append(out.detach().cpu().numpy())

    plt.plot(m_list, loss)
    plt.xlabel("Theta")
    plt.ylabel("Sinkhorn Distance")
    plt.title("Uniform Example")
    fig_name = "plots/uniform_example/" + filename + ".png"

    Path(fig_name).parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(fig_name)

    plt.show()

    df = pd.DataFrame({"theta": m_list, "sink_dist": loss})
    data_name = "data/uniform_example/" + filename + ".csv"
    Path(data_name).parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(data_name)


def gaussian_example(batch_size=100, reg=10, dim=10, filename="gaussian_example1", device="cpu"):
    m_list = range(21)
    loss = []
    for mu in m_list:
        x = np.random.normal(0, 1, (batch_size, dim))
        y = np.random.normal(mu, 1, (batch_size, dim))

        x = torch.from_numpy(x).float().to(device)
        y = torch.from_numpy(y).float().to(device)

        M = dmat(x, y)
        out = sink(M, reg=reg)
        out.backward()
        loss.append(out.detach().cpu().numpy())

    plt.plot(m_list, loss)
    plt.xlabel("Mu")
    plt.ylabel("Sinkhorn Distance")
    plt.title("Gaussian Example (Dim = " + str(dim) + ")")
    fig_name = "plots/gaussian_example/" + filename + ".png"
    Path(fig_name).parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(fig_name)
    plt.show()

    df = pd.DataFrame({"mu": m_list, "sink_dist": loss})
    data_name = "data/gaussian_example/" + filename + ".csv"
    Path(data_name).parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(data_name)


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    uniform_stabilized_example(filename="uniform_stabilized_example1", device=device)
    uniform_example(filename="uniform_example2", device=device)
    gaussian_example(reg=10000, dim=700, filename="gaussian_example3", device=device)
