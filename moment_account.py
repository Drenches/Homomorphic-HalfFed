from opacus.accountants import RDPAccountant
import pdb

def calculate_epsilon(q, sigma, delta, steps):
    accountant = RDPAccountant()
    for _ in range(steps):
        accountant.step(noise_multiplier=sigma, sample_rate=q)
        epsilon, _ = accountant.get_privacy_spent(delta=delta)
        print(epsilon)
    return epsilon

# 参数设置
q = 16/600        # 采样率
sigma = 1     # 噪声标准差
max_norm = 1
delta = 1e-5    # 隐私损失
steps = 100000    # 训练步骤数

epsilon = calculate_epsilon(q, sigma*max_norm, delta, steps)
print(f"After {steps} steps, the privacy budget is (ε = {epsilon:.2f}, δ = {delta})")


