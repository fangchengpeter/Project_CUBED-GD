import os
import torch
import matplotlib.pyplot as plt

moderate_b0_j5 = "results_trimmed_mean_random_moderate_j1"
moderate_b1_j5 = "results_trimmed_mean_random_moderate_j2"
moderate_b2_j5 = "results_trimmed_mean_random_moderate_j5"
moderate_b4_j5 = "results_trimmed_mean_random_moderate_j10"
def plot():
    b0_acc = os.path.join(moderate_b0_j5, "mean_accuracy.pt")
    b1_acc = os.path.join(moderate_b1_j5, "mean_accuracy.pt")
    b2_acc = os.path.join(moderate_b2_j5, "mean_accuracy.pt")
    b4_acc = os.path.join(moderate_b4_j5, "mean_accuracy.pt")



    plt.figure(figsize=(8, 5))
    plt.plot(torch.load(b0_acc), label="j1")
    plt.plot(torch.load(b1_acc), label="j2")
    plt.plot(torch.load(b2_acc), label="j5")
    plt.plot(torch.load(b4_acc), label="j10")


    plt.xlabel('Iterations')
    plt.ylabel('Mean Accuracy')
    plt.title('CIFAR10: moderate non iid (different j)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("moderate non iid different j")
    plt.show()
    

if __name__ == "__main__":
    plot()