import torch
import os
target_norm = 20
# 加载旧权重
state = torch.load('model_0.pt')
print(state)
W_old = state['fc.weight'].cpu()
W_old_flat = W_old.view(-1)
u = W_old_flat / W_old_flat.norm()  # 单位向量

# 随机初始化
W_new_flat = torch.randn_like(W_old_flat)

# 1. 去除与旧权重的重合部分
proj1 = (W_new_flat @ u) * u
W_new_flat = W_new_flat - proj1

# 2. 去除与 all-ones 的重合部分（可选）
ones_flat = torch.ones_like(W_new_flat)
ones_flat = ones_flat / ones_flat.norm()
proj2 = (W_new_flat @ ones_flat) * ones_flat
W_new_flat = W_new_flat - proj2

# 
W_new_flat = W_new_flat / W_new_flat.norm() * target_norm


# 恢复 shape
W_new = W_new_flat.view_as(W_old)

print(f"Dot with W_old: {(W_new_flat @ u).item():.6e}")
print(f"Dot with all-ones: {(W_new_flat @ ones_flat).item():.6e}")
print(f"Norm of new W: {W_new.norm().item():.4f}")
torch.save(W_new, "new_init_weight.pt")

