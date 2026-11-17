from __future__ import annotations

import math
import time
from typing import Optional, Dict, Tuple

import numpy as np
import numba
from numba import cuda
from numba.cuda.random import (
    create_xoroshiro128p_states,
    xoroshiro128p_normal_float64,
    xoroshiro128p_uniform_float64,
)


# ==============================================================================
# PART 1: CUDA GPU 核函数
# AI 诱发技术债务双策略空间公共物品博弈模型
# ==============================================================================


@cuda.jit(device=True)
def _clip01(x):
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


@cuda.jit(device=True)
def _safe_exp(x):
    if x > 50.0:
        return math.exp(50.0)
    if x < -50.0:
        return math.exp(-50.0)
    return math.exp(x)


@cuda.jit(device=True)
def _sigmoid_diff(diff, temperature):
    """
    返回 1 / (1 + exp(-diff / temperature))。
    使用截断避免 exp 溢出。
    """
    if temperature <= 0.0:
        if diff > 0.0:
            return 1.0
        elif diff < 0.0:
            return 0.0
        else:
            return 0.5

    z = diff / temperature
    if z > 20.0:
        return 1.0
    if z < -20.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-z))


@cuda.jit
def update_debt_kernel(
    x_grid,
    a_grid,
    D_grid,
    D_next,
    grid_size,
    delta_D,
    beta_a,
    omega_D,
    beta_N,
    mu_x,
    kappa_D,
    spillover_include_self,
):
    """
    AI 诱发技术债务动态：

    D_i(t+1) = clip[
        (1-delta_D)D_i(t)
        + beta_a a_i(t)(1-x_i(t))(1+omega_D D_i(t))
        + beta_N * local_D_i(t)
        - mu_x x_i(t) D_i(t)/(D_i(t)+kappa_D)
    ]

    本版本删除可靠性记忆 M_i(t) 的直接清债项；
    M_i(t) 仅在记忆更新、亲和度评估与免疫刺激中发挥作用。

    spillover_include_self = 1: local_D 使用自身 + 四邻居平均；
    spillover_include_self = 0: local_D 只使用四邻居平均。
    """
    r, c = cuda.grid(2)
    if r >= grid_size or c >= grid_size:
        return

    offsets_r = cuda.local.array(5, numba.int32)
    offsets_c = cuda.local.array(5, numba.int32)
    offsets_r[0] = 0
    offsets_c[0] = 0
    offsets_r[1] = -1
    offsets_c[1] = 0
    offsets_r[2] = 1
    offsets_c[2] = 0
    offsets_r[3] = 0
    offsets_c[3] = -1
    offsets_r[4] = 0
    offsets_c[4] = 1

    local_sum_D = 0.0
    local_count = 0.0

    start_idx = 0
    if spillover_include_self == 0:
        start_idx = 1

    for k in range(start_idx, 5):
        nr = (r + offsets_r[k] + grid_size) % grid_size
        nc = (c + offsets_c[k] + grid_size) % grid_size
        local_sum_D += D_grid[nr, nc]
        local_count += 1.0

    local_avg_D = local_sum_D / local_count

    x_i = x_grid[r, c]
    a_i = a_grid[r, c]
    D_i = D_grid[r, c]

    residual = (1.0 - delta_D) * D_i
    ai_induced_debt = beta_a * a_i * (1.0 - x_i) * (1.0 + omega_D * D_i)
    spillover = beta_N * local_avg_D

    if D_i + kappa_D > 0.0:
        qa_cleaning = mu_x * x_i * (D_i / (D_i + kappa_D))
    else:
        qa_cleaning = 0.0

    value = residual + ai_induced_debt + spillover - qa_cleaning
    D_next[r, c] = _clip01(value)


@cuda.jit
def calc_payoffs_kernel(
    x_grid,
    a_grid,
    D_next,
    payoffs,
    grid_size,
    r_public,
    theta_D,
    b_ai,
    A_env,
    chi_x,
    c_x,
    c_a,
    tau_D,
):
    """
    个体综合收益：

    Pi_i(t) = average_g B_g(t)
              + b_ai A_env a_i(t)(1-chi_x x_i(t))
              - c_x x_i(t)
              - c_a a_i(t)
              - tau_D D_i(t+1)^2

    B_g(t) = r_public * avg_x_g(t) * exp[-theta_D * avg_D_g(t+1)]

    每个个体参与 5 个重叠的冯·诺依曼公共物品群体。
    """
    r, c = cuda.grid(2)
    if r >= grid_size or c >= grid_size:
        return

    offsets_r = cuda.local.array(5, numba.int32)
    offsets_c = cuda.local.array(5, numba.int32)
    offsets_r[0] = 0
    offsets_c[0] = 0
    offsets_r[1] = -1
    offsets_c[1] = 0
    offsets_r[2] = 1
    offsets_c[2] = 0
    offsets_r[3] = 0
    offsets_c[3] = -1
    offsets_r[4] = 0
    offsets_c[4] = 1

    public_benefit_sum = 0.0

    # 个体 (r,c) 属于以自身和四邻居为中心的 5 个群体。
    for g in range(5):
        center_r = (r - offsets_r[g] + grid_size) % grid_size
        center_c = (c - offsets_c[g] + grid_size) % grid_size

        sum_x = 0.0
        sum_D = 0.0

        for m in range(5):
            mr = (center_r + offsets_r[m] + grid_size) % grid_size
            mc = (center_c + offsets_c[m] + grid_size) % grid_size
            sum_x += x_grid[mr, mc]
            sum_D += D_next[mr, mc]

        avg_x = sum_x / 5.0
        avg_D = sum_D / 5.0
        B_g = r_public * avg_x * math.exp(-theta_D * avg_D)
        public_benefit_sum += B_g

    avg_public_benefit = public_benefit_sum / 5.0

    x_i = x_grid[r, c]
    a_i = a_grid[r, c]
    D_i_next = D_next[r, c]

    ai_short_term_gain = b_ai * A_env * a_i * (1.0 - chi_x * x_i)
    qa_cost = c_x * x_i
    ai_cost = c_a * a_i
    debt_interest = tau_D * D_i_next * D_i_next

    payoffs[r, c] = avg_public_benefit + ai_short_term_gain - qa_cost - ai_cost - debt_interest


@cuda.jit
def update_memory_kernel(
    x_grid,
    D_next,
    M_grid,
    M_next,
    grid_size,
    lambda_M,
    u_D,
):
    """
    可靠性记忆：

    q_i(t) = x_i(t) exp[-u_D D_i(t+1)]
    M_i(t+1) = (1-lambda_M)M_i(t) + lambda_M q_i(t)
    """
    r, c = cuda.grid(2)
    if r >= grid_size or c >= grid_size:
        return

    x_i = x_grid[r, c]
    D_i_next = D_next[r, c]
    M_i = M_grid[r, c]

    q_i = x_i * math.exp(-u_D * D_i_next)
    value = (1.0 - lambda_M) * M_i + lambda_M * q_i
    M_next[r, c] = _clip01(value)


@cuda.jit
def calc_concentrations_kernel(
    x_grid,
    a_grid,
    concentrations,
    grid_size,
    delta,
    lambda_a,
    concentration_include_self,
):
    """
    双策略局部浓度：

    rho_j(t) = average_k exp(-[(x_k-x_j)^2 + lambda_a(a_k-a_j)^2] / [2 sigma_C^2])

    concentration_include_self = 1: 自身 + 四邻居；
    concentration_include_self = 0: 只使用四邻居。
    """
    r, c = cuda.grid(2)
    if r >= grid_size or c >= grid_size:
        return

    offsets_r = cuda.local.array(5, numba.int32)
    offsets_c = cuda.local.array(5, numba.int32)
    offsets_r[0] = 0
    offsets_c[0] = 0
    offsets_r[1] = -1
    offsets_c[1] = 0
    offsets_r[2] = 1
    offsets_c[2] = 0
    offsets_r[3] = 0
    offsets_c[3] = -1
    offsets_r[4] = 0
    offsets_c[4] = 1

    x_j = x_grid[r, c]
    a_j = a_grid[r, c]

    start_idx = 0
    if concentration_include_self == 0:
        start_idx = 1

    sum_sim = 0.0
    count = 0.0
    denom = delta

    for k in range(start_idx, 5):
        nr = (r + offsets_r[k] + grid_size) % grid_size
        nc = (c + offsets_c[k] + grid_size) % grid_size

        dx = x_grid[nr, nc] - x_j
        da = a_grid[nr, nc] - a_j
        dist_sq = dx * dx + lambda_a * da * da

        if denom <= 0.0:
            if dist_sq == 0.0:
                sim = 1.0
            else:
                sim = 0.0
        else:
            sim = math.exp(-dist_sq / denom)

        sum_sim += sim
        count += 1.0

    concentrations[r, c] = sum_sim / count


@cuda.jit
def immune_sweep_kernel(
    rng_states,
    x_grid,
    a_grid,
    M_next,
    payoffs,
    concentrations,
    new_x,
    new_a,
    grid_size,
    w_payoff,
    w_memory,
    K_P,
    K_M,
    global_avg_M,
    alpha_conc,
    screening_mode,
    selection_rule,
    fermi_K,
    sigma_min,
    sigma_max,
    learning_rate,
):
    """
    MDAE 亲和度 + 浓度抑制 + 自适应扰动 + 惯性策略更新。

    Aff_{i->j} = w_payoff P'_{j|i} + w_memory M'_j

    Stim_{i->j} = Aff_{i->j} * exp[-alpha_conc * z_j]

    ``screening_mode`` is retained as a backward-compatible parameter name,
    but now selects the concentration-regulation signal z_j:
        0: no concentration regulation, z_j = 0
        1: reliability gate without spatial concentration, z_j = 1 - M'_j
        2: unconditioned concentration diagnostic, z_j = rho_j
        3: reliability-conditioned concentration (default),
           z_j = rho_j * (1 - M'_j)

    Technical debt is deliberately absent from z_j.  It remains endogenous in
    debt accumulation, payoff evaluation, and reliability-memory updating.

    selection_rule = 0: 个体 i 选择 Stim 最大的候选 j。
    selection_rule = 1: 随机选择一个邻居 j，并根据传统收益费米概率
        1 / (1 + exp[-(Pi_j-Pi_i)/K_F]) 决定是否采用该邻居；否则保留自身。
    候选模板确定后，在其策略附近扰动并进行惯性更新。
    """
    r, c = cuda.grid(2)
    if r >= grid_size or c >= grid_size:
        return

    offsets_r = cuda.local.array(5, numba.int32)
    offsets_c = cuda.local.array(5, numba.int32)
    offsets_r[0] = 0
    offsets_c[0] = 0
    offsets_r[1] = -1
    offsets_c[1] = 0
    offsets_r[2] = 1
    offsets_c[2] = 0
    offsets_r[3] = 0
    offsets_c[3] = -1
    offsets_r[4] = 0
    offsets_c[4] = 1

    cand_payoffs = cuda.local.array(5, numba.float64)
    cand_memory = cuda.local.array(5, numba.float64)
    cand_conc = cuda.local.array(5, numba.float64)
    cand_aff = cuda.local.array(5, numba.float64)

    sum_P = 0.0

    for idx in range(5):
        nr = (r + offsets_r[idx] + grid_size) % grid_size
        nc = (c + offsets_c[idx] + grid_size) % grid_size

        P_j = payoffs[nr, nc]
        M_j = M_next[nr, nc]
        rho_j = concentrations[nr, nc]

        cand_payoffs[idx] = P_j
        cand_memory[idx] = M_j
        cand_conc[idx] = rho_j

        sum_P += P_j

    local_avg_P = sum_P / 5.0

    flat_idx = r * grid_size + c
    best_idx = 0
    best_stim = -1.0

    for idx in range(5):
        P_norm = _sigmoid_diff(cand_payoffs[idx] - local_avg_P, K_P)
        M_norm = _sigmoid_diff(cand_memory[idx] - global_avg_M, K_M)

        affinity = w_payoff * P_norm + w_memory * M_norm
        affinity = _clip01(affinity)
        cand_aff[idx] = affinity

        if screening_mode == 0:
            regulation_signal = 0.0
        elif screening_mode == 1:
            regulation_signal = 1.0 - M_norm
        elif screening_mode == 2:
            regulation_signal = cand_conc[idx]
        else:
            regulation_signal = cand_conc[idx] * (1.0 - M_norm)

        penalty = alpha_conc * regulation_signal
        stim = affinity * math.exp(-penalty)

        if stim > best_stim:
            best_stim = stim
            best_idx = idx

    if selection_rule == 1:
        # Traditional pairwise Fermi imitation based only on payoff difference.
        # Draw uniformly from the four graph neighbors; index 0 is the focal unit.
        neighbor_draw = xoroshiro128p_uniform_float64(rng_states, flat_idx)
        neighbor_idx = 1 + int(neighbor_draw * 4.0)
        if neighbor_idx > 4:
            neighbor_idx = 4

        adoption_probability = _sigmoid_diff(
            cand_payoffs[neighbor_idx] - cand_payoffs[0],
            fermi_K,
        )
        adoption_draw = xoroshiro128p_uniform_float64(rng_states, flat_idx)
        if adoption_draw < adoption_probability:
            best_idx = neighbor_idx
        else:
            best_idx = 0

    parent_r = (r + offsets_r[best_idx] + grid_size) % grid_size
    parent_c = (c + offsets_c[best_idx] + grid_size) % grid_size
    parent_affinity = cand_aff[best_idx]

    # 与模型定义一致：亲和度越高，扰动越小。
    sigma = sigma_min + (sigma_max - sigma_min) * (1.0 - parent_affinity)
    if sigma < 0.0:
        sigma = 0.0

    noise_x = xoroshiro128p_normal_float64(rng_states, flat_idx) * sigma
    noise_a = xoroshiro128p_normal_float64(rng_states, flat_idx) * sigma

    target_x = _clip01(x_grid[parent_r, parent_c] + noise_x)
    target_a = _clip01(a_grid[parent_r, parent_c] + noise_a)

    old_x = x_grid[r, c]
    old_a = a_grid[r, c]

    updated_x = (1.0 - learning_rate) * old_x + learning_rate * target_x
    updated_a = (1.0 - learning_rate) * old_a + learning_rate * target_a

    new_x[r, c] = _clip01(updated_x)
    new_a[r, c] = _clip01(updated_a)


# ==============================================================================
# PART 2: GPU 模型控制器类
# 只包含模型定义、状态推进和状态读取；不包含实验、画图、参数扫描或稳态统计。
# ==============================================================================


class AITechnicalDebtMDAE_GPU:
    """
    AI 诱发技术债务的双策略空间公共物品博弈 GPU 模型。

    状态变量：
        x_grid: 质量保障投入，范围 [0, 1]
        a_grid: 个体 AI 使用强度，范围 [0, 1]
        D_grid: AI 诱发技术债务，范围 [0, 1]
        M_grid: 可靠性记忆，范围 [0, 1]

    外部环境参数：
        A_env: 外部 AI 环境强度，范围 [0, 1]

    默认初始化：
        x_i(0) 随机异质，但系统总平均严格为 0.5；
        a_i(0) 随机异质，但系统总平均严格为 0.5；
        D_i(0) 为低初始技术债，默认 D_i(0) ~ U(0, 0.02)；
        M_i(0) = 0.5，表示无历史信息时的中性可靠性记忆。

    典型实验脚本调用方式：
        model = AITechnicalDebtMDAE_GPU(grid_size=100, seed=1)
        model.evolve(steps=1000)
        state = model.get_state()

    注意：本类不包含任何实验逻辑，例如参数扫描、稳态平均、绘图或文件保存。
    """

    def __init__(
        self,
        grid_size: int = 100,
        seed: Optional[int] = None,
        # 外部 AI 环境
        A_env: float = 0.5,
        # 技术债动态参数
        delta_D: float = 0.1,
        beta_a: float = 0.5,
        omega_D: float = 0.5,
        beta_N: float = 0.1,
        mu_x: float = 0.3,
        kappa_D: float = 0.1,
        spillover_include_self: bool = True,
        # 公共物品收益参数
        r_public: float = 3.5,
        theta_D: float = 1.0,
        b_ai: float = 1.0,
        chi_x: float = 0.3,
        c_x: float = 1.0,
        c_a: float = 0.1,
        tau_D: float = 1.0,
        # 可靠性记忆参数
        lambda_0: float = 0.1,
        lambda_A: float = 0.4,
        u_D: float = 2.0,
        # MDAE 亲和度参数
        w_payoff: float = 0.7,
        w_memory: float = 0.3,
        K_P: float = 0.2,
        K_M: float = 0.2,
        # 双策略浓度与免疫抑制参数
        delta: float = 0.3,
        lambda_a: float = 1.0,
        concentration_include_self: bool = False,
        alpha_conc: float = 3.0,
        screening_mode: int = 3,
        # 0: deterministic maximum-stimulus selection; 1: payoff-based Fermi imitation
        selection_rule: int = 0,
        fermi_K: float = 0.1,
        # 自适应扰动与惯性更新参数
        sigma_min: float = 0.001,
        sigma_max: float = 0.03,
        learning_rate: float = 0.5,
        # 初始状态，可选
        init_x: Optional[np.ndarray] = None,
        init_a: Optional[np.ndarray] = None,
        init_D: Optional[np.ndarray] = None,
        init_M: Optional[np.ndarray] = None,
        # 当 init_D=None 时，默认 D_i(0) ~ U(0, init_D_max)。
        # 若希望严格 D_i(0)=0，可设置 init_D_max=0.0。
        init_D_max: float = 0.02,
        # CUDA 配置
        threadsperblock: Tuple[int, int] = (8, 8),
    ):
        self.grid_size = int(grid_size)
        self.seed = int(time.time()) if seed is None else int(seed)

        self.A_env = float(A_env)

        self.delta_D = float(delta_D)
        self.beta_a = float(beta_a)
        self.omega_D = float(omega_D)
        self.beta_N = float(beta_N)
        self.mu_x = float(mu_x)
        self.kappa_D = float(kappa_D)
        self.spillover_include_self = 1 if spillover_include_self else 0

        self.r_public = float(r_public)
        self.theta_D = float(theta_D)
        self.b_ai = float(b_ai)
        self.chi_x = float(chi_x)
        self.c_x = float(c_x)
        self.c_a = float(c_a)
        self.tau_D = float(tau_D)

        self.lambda_0 = float(lambda_0)
        self.lambda_A = float(lambda_A)
        self.u_D = float(u_D)

        weight_sum = float(w_payoff) + float(w_memory)
        if weight_sum <= 0.0:
            self.w_payoff = 0.5
            self.w_memory = 0.5
        else:
            self.w_payoff = float(w_payoff) / weight_sum
            self.w_memory = float(w_memory) / weight_sum

        self.K_P = float(K_P)
        self.K_M = float(K_M)

        self.delta = float(delta)
        self.lambda_a = float(lambda_a)
        self.concentration_include_self = 1 if concentration_include_self else 0
        self.alpha_conc = float(alpha_conc)
        self.screening_mode = int(screening_mode)
        self.selection_rule = int(selection_rule)
        self.fermi_K = float(fermi_K)

        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.learning_rate = float(learning_rate)

        self.init_D_max = float(init_D_max)

        self.threadsperblock = threadsperblock
        blocks_x = math.ceil(self.grid_size / self.threadsperblock[0])
        blocks_y = math.ceil(self.grid_size / self.threadsperblock[1])
        self.blockspergrid = (blocks_x, blocks_y)

        self._validate_parameters()
        self._init_device_arrays(init_x=init_x, init_a=init_a, init_D=init_D, init_M=init_M)

        self.iteration = 0

    def _validate_parameters(self) -> None:
        if self.grid_size <= 0:
            raise ValueError("grid_size must be positive.")
        if not (0.0 <= self.A_env <= 1.0):
            raise ValueError("A_env must be in [0, 1].")
        if self.kappa_D < 0.0:
            raise ValueError("kappa_D must be non-negative.")
        if self.delta < 0.0:
            raise ValueError("delta must be non-negative.")
        if self.screening_mode not in (0, 1, 2, 3):
            raise ValueError(
                "screening_mode must be 0 (none), 1 (1-M'), 2 (rho), "
                "or 3 (rho*(1-M'))."
            )
        if self.selection_rule not in (0, 1):
            raise ValueError("selection_rule must be 0 (argmax) or 1 (payoff Fermi).")
        if self.fermi_K <= 0.0:
            raise ValueError("fermi_K must be positive.")
        if self.sigma_min < 0.0 or self.sigma_max < 0.0:
            raise ValueError("sigma_min and sigma_max must be non-negative.")
        if self.sigma_min > self.sigma_max:
            raise ValueError("sigma_min cannot be larger than sigma_max.")
        if not (0.0 <= self.learning_rate <= 1.0):
            raise ValueError("learning_rate must be in [0, 1].")
        if not (0.0 <= self.init_D_max <= 1.0):
            raise ValueError("init_D_max must be in [0, 1].")

    def _lambda_M(self) -> float:
        value = self.lambda_0 + self.lambda_A * self.A_env
        if value < 0.0:
            return 0.0
        if value > 1.0:
            return 1.0
        return value

    def _random_array_with_exact_mean_half(self, rng: np.random.Generator) -> np.ndarray:
        """
        生成 [0,1] 随机矩阵，并保证系统总平均严格为 0.5。

        方法：
            成对生成 u 和 1-u。
            如果格点总数为奇数，最后补一个 0.5。
        """
        shape = (self.grid_size, self.grid_size)
        total_size = self.grid_size * self.grid_size
        half_size = total_size // 2

        first_half = rng.random(half_size, dtype=np.float64)

        values = np.empty(total_size, dtype=np.float64)
        values[:half_size] = first_half
        values[half_size:2 * half_size] = 1.0 - first_half

        if total_size % 2 == 1:
            values[-1] = 0.5

        rng.shuffle(values)
        return np.ascontiguousarray(values.reshape(shape))

    def _prepare_init_array(
        self,
        arr: Optional[np.ndarray],
        rng: np.random.Generator,
        fill: Optional[float] = None,
        exact_mean_half: bool = False,
        low_debt: bool = False,
    ) -> np.ndarray:
        """
        准备初始数组。

        exact_mean_half=True:
            当 arr=None 时，生成均值严格为 0.5 的随机矩阵。
            用于 x_i(0), a_i(0)。

        low_debt=True:
            当 arr=None 时，生成低初始技术债：
                D_i(0) ~ U(0, init_D_max)
            用于 D_i(0)。

        fill:
            当 arr=None 时，生成常数矩阵。
            用于 M_i(0)=0.5。
        """
        shape = (self.grid_size, self.grid_size)

        if arr is None:
            if exact_mean_half:
                out = self._random_array_with_exact_mean_half(rng)
            elif low_debt:
                if self.init_D_max == 0.0:
                    out = np.zeros(shape, dtype=np.float64)
                else:
                    out = rng.random(shape, dtype=np.float64) * self.init_D_max
            elif fill is not None:
                out = np.full(shape, fill, dtype=np.float64)
            else:
                out = rng.random(shape, dtype=np.float64)
        else:
            out = np.asarray(arr, dtype=np.float64)
            if out.shape != shape:
                raise ValueError(f"Initial array shape must be {shape}, got {out.shape}.")
            out = np.clip(out, 0.0, 1.0)

        return np.ascontiguousarray(out)

    def _init_device_arrays(
        self,
        init_x: Optional[np.ndarray],
        init_a: Optional[np.ndarray],
        init_D: Optional[np.ndarray],
        init_M: Optional[np.ndarray],
    ) -> None:
        rng = np.random.default_rng(self.seed)

        # x_i(0) 和 a_i(0) 使用随机异质初始化，
        # 但系统总平均严格为 0.5。
        x_host = self._prepare_init_array(init_x, rng, exact_mean_half=True)
        a_host = self._prepare_init_array(init_a, rng, exact_mean_half=True)

        # D_i(0) 表示技术债初始水平。
        # 技术债应由后续 AI 使用与质量保障不足共同诱发，
        # 因此默认设为低初始债务，而不是 U(0,1)。
        D_host = self._prepare_init_array(init_D, rng, low_debt=True)

        # M_i(0) 是可靠性记忆/声誉的中性先验。
        # 不建议设为接近 0，否则相当于所有个体初始低声誉。
        M_host = self._prepare_init_array(init_M, rng, fill=0.5)

        self.d_x = cuda.to_device(x_host)
        self.d_a = cuda.to_device(a_host)
        self.d_D = cuda.to_device(D_host)
        self.d_M = cuda.to_device(M_host)

        shape = (self.grid_size, self.grid_size)
        self.d_D_next = cuda.device_array(shape, dtype=np.float64)
        self.d_M_next = cuda.device_array(shape, dtype=np.float64)
        self.d_payoff = cuda.device_array(shape, dtype=np.float64)
        self.d_concentration = cuda.device_array(shape, dtype=np.float64)
        self.d_new_x = cuda.device_array(shape, dtype=np.float64)
        self.d_new_a = cuda.device_array(shape, dtype=np.float64)

        self.rng_states = create_xoroshiro128p_states(self.grid_size * self.grid_size, seed=self.seed)

    def set_ai_environment(self, A_env: float) -> None:
        """修改外部 AI 环境强度 A_env。"""
        if not (0.0 <= A_env <= 1.0):
            raise ValueError("A_env must be in [0, 1].")
        self.A_env = float(A_env)

    def reset(
        self,
        seed: Optional[int] = None,
        init_x: Optional[np.ndarray] = None,
        init_a: Optional[np.ndarray] = None,
        init_D: Optional[np.ndarray] = None,
        init_M: Optional[np.ndarray] = None,
        init_D_max: Optional[float] = None,
    ) -> None:
        """
        重置模型状态。

        如果 init_D_max 不为 None，则同时更新默认初始技术债上界。
        """
        if seed is not None:
            self.seed = int(seed)
        if init_D_max is not None:
            self.init_D_max = float(init_D_max)
            self._validate_parameters()
        self._init_device_arrays(init_x=init_x, init_a=init_a, init_D=init_D, init_M=init_M)
        self.iteration = 0

    def step(self, synchronize: bool = False) -> None:
        """
        推进一轮同步演化：
            1. 更新技术债 D(t+1)
            2. 计算收益 Pi(t)
            3. 更新可靠性记忆 M(t+1)
            4. 计算双策略浓度 rho(t)
            5. MDAE 免疫选择与策略更新 s(t+1)
            6. 交换 GPU 状态指针
        """
        lambda_M = self._lambda_M()

        update_debt_kernel[self.blockspergrid, self.threadsperblock](
            self.d_x,
            self.d_a,
            self.d_D,
            self.d_D_next,
            self.grid_size,
            self.delta_D,
            self.beta_a,
            self.omega_D,
            self.beta_N,
            self.mu_x,
            self.kappa_D,
            self.spillover_include_self,
        )

        calc_payoffs_kernel[self.blockspergrid, self.threadsperblock](
            self.d_x,
            self.d_a,
            self.d_D_next,
            self.d_payoff,
            self.grid_size,
            self.r_public,
            self.theta_D,
            self.b_ai,
            self.A_env,
            self.chi_x,
            self.c_x,
            self.c_a,
            self.tau_D,
        )

        update_memory_kernel[self.blockspergrid, self.threadsperblock](
            self.d_x,
            self.d_D_next,
            self.d_M,
            self.d_M_next,
            self.grid_size,
            lambda_M,
            self.u_D,
        )

        # 参考第一篇 GPU 代码的写法：将全局均值提取到 CPU 作为免疫选择的全局基准。
        global_avg_M = float(np.mean(self.d_M_next.copy_to_host()))

        calc_concentrations_kernel[self.blockspergrid, self.threadsperblock](
            self.d_x,
            self.d_a,
            self.d_concentration,
            self.grid_size,
            self.delta,
            self.lambda_a,
            self.concentration_include_self,
        )

        immune_sweep_kernel[self.blockspergrid, self.threadsperblock](
            self.rng_states,
            self.d_x,
            self.d_a,
            self.d_M_next,
            self.d_payoff,
            self.d_concentration,
            self.d_new_x,
            self.d_new_a,
            self.grid_size,
            self.w_payoff,
            self.w_memory,
            self.K_P,
            self.K_M,
            global_avg_M,
            self.alpha_conc,
            self.screening_mode,
            self.selection_rule,
            self.fermi_K,
            self.sigma_min,
            self.sigma_max,
            self.learning_rate,
        )

        self.d_x, self.d_new_x = self.d_new_x, self.d_x
        self.d_a, self.d_new_a = self.d_new_a, self.d_a
        self.d_D, self.d_D_next = self.d_D_next, self.d_D
        self.d_M, self.d_M_next = self.d_M_next, self.d_M

        self.iteration += 1

        if synchronize:
            cuda.synchronize()

    def evolve(self, steps: int, synchronize: bool = False) -> None:
        """连续推进若干轮。该方法只推进模型，不计算稳态均值或实验指标。"""
        for _ in range(int(steps)):
            self.step(synchronize=False)
        if synchronize:
            cuda.synchronize()

    def get_state(self) -> Dict[str, np.ndarray]:
        """
        读取当前状态到 CPU。

        返回：
            x: 质量保障投入矩阵
            a: AI 使用强度矩阵
            D: 技术债矩阵
            M: 可靠性记忆矩阵
        """
        return {
            "x": self.d_x.copy_to_host(),
            "a": self.d_a.copy_to_host(),
            "D": self.d_D.copy_to_host(),
            "M": self.d_M.copy_to_host(),
        }

    def get_observables(self) -> Dict[str, np.ndarray]:
        """
        读取辅助观测量。

        注意：payoff 和 concentration 对应最近一次 step 中使用的状态。
        如果刚初始化后尚未调用 step，则这些数组尚未被模型更新。
        """
        return {
            "payoff": self.d_payoff.copy_to_host(),
            "concentration": self.d_concentration.copy_to_host(),
        }

    def get_means(self) -> Dict[str, float]:
        """
        返回当前状态均值。该函数只是读取模型状态，
        不执行稳态统计、参数扫描或实验聚合。
        """
        state = self.get_state()
        return {
            "mean_x": float(np.mean(state["x"])),
            "mean_a": float(np.mean(state["a"])),
            "mean_D": float(np.mean(state["D"])),
            "mean_M": float(np.mean(state["M"])),
        }

    def device_arrays(self) -> Dict[str, cuda.devicearray.DeviceNDArray]:
        """
        返回 GPU 端数组引用，便于高级实验脚本直接调用。
        """
        return {
            "x": self.d_x,
            "a": self.d_a,
            "D": self.d_D,
            "M": self.d_M,
            "payoff": self.d_payoff,
            "concentration": self.d_concentration,
        }
