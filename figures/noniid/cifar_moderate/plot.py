import os
import torch
import matplotlib.pyplot as plt

moderate_b0_j5 = "results_trimmed_mean_random_moderate_j5_b0"
moderate_b1_j5 = "results_trimmed_mean_random_moderate_j5_b1"
moderate_b2_j5 = "results_trimmed_mean_random_moderate_j5_b2"
moderate_b4_j5 = "results_trimmed_mean_random_moderate_j5_b4"
def plot():
    b0_acc = os.path.join(moderate_b0_j5, "mean_accuracy.pt")
    b1_acc = os.path.join(moderate_b1_j5, "mean_accuracy.pt")
    b2_acc = os.path.join(moderate_b2_j5, "mean_accuracy.pt")
    b4_acc = os.path.join(moderate_b4_j5, "mean_accuracy.pt")



    plt.figure(figsize=(8, 5))
    plt.plot(torch.load(b0_acc), label="b0")
    plt.plot(torch.load(b1_acc), label="b1")
    plt.plot(torch.load(b2_acc), label="b2")
    plt.plot(torch.load(b4_acc), label="b4")


    plt.xlabel('Iterations')
    plt.ylabel('Mean Accuracy')
    plt.title('CIFAR10: moderate non iid (j=5)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("moderate non iid j5")
    plt.show()
    

if __name__ == "__main__":
    plot()