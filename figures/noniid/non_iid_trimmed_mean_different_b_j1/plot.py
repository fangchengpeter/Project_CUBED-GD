import os
import torch
import matplotlib.pyplot as plt

b0 = "results_trimmed_mean_random_b0"
b1 = "results_trimmed_mean_random_b1"
b2 = "results_trimmed_mean_random_b2"
b4 = "results_trimmed_mean_random_b4"
b8 = "results_trimmed_mean_random_b8"

def moving_avg(x, k=10):
    return torch.tensor([x[i:i+k].mean().item() for i in range(len(x)-k+1)])

def plot():
    b0_acc = torch.load(os.path.join(b0, "mean_accuracy.pt"))
    b1_acc = torch.load(os.path.join(b1, "mean_accuracy.pt"))
    b2_acc = torch.load(os.path.join(b2, "mean_accuracy.pt"))
    b4_acc = torch.load(os.path.join(b4, "mean_accuracy.pt"))


    plt.figure(figsize=(10, 6))
    plt.plot(moving_avg(b0_acc))
    plt.plot(moving_avg(b1_acc))
    plt.plot(moving_avg(b2_acc))
    plt.plot(moving_avg(b4_acc))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("non_iid_j1_different_b.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    plot()
    print("plotting")
