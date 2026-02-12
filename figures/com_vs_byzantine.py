import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

mnist = "mnist_defense_vs_non_defense.png"   # <- rename files to avoid spaces
cifar = "cifar_defense_vs_non_defense.png"

def combine_matplotlib(out="COMBRIDGE_vs_Byzantine.png"):
    img1 = mpimg.imread(mnist)
    img2 = mpimg.imread(cifar)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=300)

    # 左边图
    axes[0].imshow(img1)
    axes[0].set_title("MNIST", fontsize=14)
    axes[0].axis("off")

    # 右边图
    axes[1].imshow(img2)
    axes[1].set_title("CIFAR-10", fontsize=14)
    axes[1].axis("off")

    # 全局标签
    fig.text(0.5, 0.02, "Communication Rounds", ha="center", fontsize=16)
    fig.text(0.015, 0.5, "Mean Accuracy", va="center", rotation="vertical", fontsize=16)


    # 调整布局，减少空白
    plt.subplots_adjust(wspace=0.01, left=0.07, right=0.93, top=0.82, bottom=0.12)

    fig.savefig(out, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

if __name__ == "__main__":
    combine_matplotlib()