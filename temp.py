import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

# -----------------------------
# 1. Data Preparation
# -----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))  # Flatten 28x28 to 784
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=1000, shuffle=False)
# -----------------------------
# 2. Model Definition
# -----------------------------
class LinearClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(784, 10)
        state_to_load = torch.load("new_init_weight.pt")  # 直接是个 tensor
        assert state_to_load.shape == self.fc.weight.shape, \
            f"Loaded weight shape {state_to_load.shape} does not match {self.fc.weight.shape}"
        self.fc.weight.data.copy_(state_to_load)
        # bias 还是默认初始化
    def forward(self, x):
        return self.fc(x)



model = LinearClassifier()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# -----------------------------
# 3. Training Loop
# -----------------------------
for epoch in range(30):
    model.train()
    total_loss = 0.0
    for x, y in train_loader:
        output = model(x)  # [B, 10]
        # target_onehot = F.one_hot(y, num_classes=10).float()  # [B, 10]
        # loss = F.mse_loss(output, target_onehot) + 0.001 * model.fc.weight.norm(2)
        loss = F.cross_entropy(output, y)  # Cross-entropy loss for classification
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)

    avg_loss = total_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1:2d} | AvgTrainLoss = {avg_loss:.4f}")

# -----------------------------
# 4. Print final parameter stats
# -----------------------------
with torch.no_grad():
    weight_mean = model.fc.weight.mean().item()
    weight_std = model.fc.weight.std().item()
    print(f"✅ Final weight mean = {weight_mean:.6f}, std = {weight_std:.6f}")
    if model.fc.bias is not None:
        bias_mean = model.fc.bias.mean().item()
        bias_std = model.fc.bias.std().item()
        print(f"✅ Final bias mean   = {bias_mean:.6f}, std = {bias_std:.6f}")

# -----------------------------
# 5. Evaluation
# -----------------------------
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for x, y in test_loader:
        output = model(x)
        pred = output.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

acc = correct / total
print(f"✅ Test Accuracy: {acc:.2%}")
