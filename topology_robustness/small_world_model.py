"""GPU model on a Watts--Strogatz small-world interaction network.

The dynamical equations and initialisation match the calibrated square-grid
model; only the sparse interaction topology is changed.
"""

from __future__ import annotations

import math
import time
from typing import Optional, Dict

import numpy as np
from numba import cuda
from numba.cuda.random import (
    create_xoroshiro128p_states,
    xoroshiro128p_normal_float64,
)


# ==============================================================================
# CUDA device utilities
# ==============================================================================


@cuda.jit(device=True)
def _clip01(x):
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


@cuda.jit(device=True)
def _sigmoid_diff(diff, temperature):
    """
    1 / (1 + exp(-diff / temperature)), with overflow protection.
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


# ==============================================================================
# CUDA kernels: sparse-network version
# State arrays are one-dimensional: x[node], a[node], D[node], M[node]
# Local group centered at g: G_g = {g} union N_g
# Individual i participates in groups centered at {i} union N_i
# ==============================================================================


@cuda.jit
def update_debt_kernel(
    x,
    a,
    D,
    D_next,
    neighbor_indptr,
    neighbor_indices,
    num_nodes,
    delta_D,
    beta_a,
    omega_D,
    beta_N,
    mu_x,
    kappa_D,
    spillover_include_self,
):
    """
    AI-induced technical debt dynamics on a sparse undirected network.

    D_i(t+1) = clip[
        (1-delta_D) D_i(t)
        + beta_a a_i(t) (1-x_i(t)) (1+omega_D D_i(t))
        + beta_N * local_avg_D_i(t)
        - mu_x x_i(t) D_i(t)/(D_i(t)+kappa_D)
    ]

    Reliability memory M does not directly clear debt.
    """
    i = cuda.grid(1)
    if i >= num_nodes:
        return

    x_i = x[i]
    a_i = a[i]
    D_i = D[i]

    local_sum_D = 0.0
    local_count = 0.0

    if spillover_include_self == 1:
        local_sum_D += D_i
        local_count += 1.0

    start = neighbor_indptr[i]
    end = neighbor_indptr[i + 1]
    for p in range(start, end):
        j = neighbor_indices[p]
        local_sum_D += D[j]
        local_count += 1.0

    if local_count > 0.0:
        local_avg_D = local_sum_D / local_count
    else:
        local_avg_D = D_i

    residual = (1.0 - delta_D) * D_i
    ai_induced_debt = beta_a * a_i * (1.0 - x_i) * (1.0 + omega_D * D_i)
    spillover = beta_N * local_avg_D

    if D_i + kappa_D > 0.0:
        qa_cleaning = mu_x * x_i * (D_i / (D_i + kappa_D))
    else:
        qa_cleaning = 0.0

    D_next[i] = _clip01(residual + ai_induced_debt + spillover - qa_cleaning)


@cuda.jit
def calc_group_benefit_kernel(
    x,
    D_next,
    group_benefit,
    neighbor_indptr,
    neighbor_indices,
    num_nodes,
    r_public,
    theta_D,
):
    """
    Compute local public goods benefit for every center group G_g.

    B_g = r_public * avg_x_g * exp(-theta_D * avg_D_g),
    where G_g = {g} union N_g.
    """
    g = cuda.grid(1)
    if g >= num_nodes:
        return

    sum_x = x[g]
    sum_D = D_next[g]
    count = 1.0

    start = neighbor_indptr[g]
    end = neighbor_indptr[g + 1]
    for p in range(start, end):
        j = neighbor_indices[p]
        sum_x += x[j]
        sum_D += D_next[j]
        count += 1.0

    avg_x = sum_x / count
    avg_D = sum_D / count

    group_benefit[g] = r_public * avg_x * math.exp(-theta_D * avg_D)


@cuda.jit
def calc_payoffs_from_group_benefit_kernel(
    x,
    a,
    D_next,
    group_benefit,
    payoffs,
    neighbor_indptr,
    neighbor_indices,
    num_nodes,
    b_ai,
    A_env,
    chi_x,
    c_x,
    c_a,
    tau_D,
):
    """
    Individual payoff on a sparse undirected network.

    Pi_i(t) = average_{g in {i} union N_i} B_g(t)
              + b_ai A_env a_i(t)(1-chi_x x_i(t))
              - c_x x_i(t)
              - c_a a_i(t)
              - tau_D D_i(t+1)^2
    """
    i = cuda.grid(1)
    if i >= num_nodes:
        return

    public_sum = group_benefit[i]
    group_count = 1.0

    start = neighbor_indptr[i]
    end = neighbor_indptr[i + 1]
    for p in range(start, end):
        g = neighbor_indices[p]
        public_sum += group_benefit[g]
        group_count += 1.0

    avg_public_benefit = public_sum / group_count

    x_i = x[i]
    a_i = a[i]
    D_i_next = D_next[i]

    ai_short_term_gain = b_ai * A_env * a_i * (1.0 - chi_x * x_i)
    qa_cost = c_x * x_i
    ai_cost = c_a * a_i
    debt_interest = tau_D * D_i_next * D_i_next

    payoffs[i] = avg_public_benefit + ai_short_term_gain - qa_cost - ai_cost - debt_interest


@cuda.jit
def update_memory_kernel(
    x,
    D_next,
    M,
    M_next,
    num_nodes,
    lambda_M,
    u_D,
):
    """
    Reliability memory dynamics.

    q_i(t) = x_i(t) exp[-u_D D_i(t+1)]
    M_i(t+1) = (1-lambda_M) M_i(t) + lambda_M q_i(t)
    """
    i = cuda.grid(1)
    if i >= num_nodes:
        return

    q_i = x[i] * math.exp(-u_D * D_next[i])
    M_next[i] = _clip01((1.0 - lambda_M) * M[i] + lambda_M * q_i)


@cuda.jit
def zero_scalar_kernel(buf):
    if cuda.grid(1) == 0:
        buf[0] = 0.0


@cuda.jit
def reduce_sum_kernel(arr, sum_buf, num_nodes):
    """
    Sum arr on GPU into sum_buf[0] using atomic add.
    The caller should zero sum_buf before launching this kernel.
    """
    i = cuda.grid(1)
    if i >= num_nodes:
        return
    cuda.atomic.add(sum_buf, 0, arr[i])


@cuda.jit
def calc_concentrations_kernel(
    x,
    a,
    concentrations,
    neighbor_indptr,
    neighbor_indices,
    num_nodes,
    delta,
    lambda_a,
    concentration_include_self,
):
    """
    Dual-strategy local concentration for each candidate node j.

    rho_j = average_{k in local set of j}
            exp(-[(x_k-x_j)^2 + lambda_a(a_k-a_j)^2] / delta)

    concentration_include_self = 1: local set = {j} union N_j
    concentration_include_self = 0: local set = N_j only
    """
    j = cuda.grid(1)
    if j >= num_nodes:
        return

    x_j = x[j]
    a_j = a[j]

    sum_sim = 0.0
    count = 0.0

    if concentration_include_self == 1:
        sum_sim += 1.0
        count += 1.0

    start = neighbor_indptr[j]
    end = neighbor_indptr[j + 1]
    for p in range(start, end):
        k = neighbor_indices[p]
        dx = x[k] - x_j
        da = a[k] - a_j
        dist_sq = dx * dx + lambda_a * da * da

        if delta <= 0.0:
            if dist_sq == 0.0:
                sim = 1.0
            else:
                sim = 0.0
        else:
            sim = math.exp(-dist_sq / delta)

        sum_sim += sim
        count += 1.0

    if count > 0.0:
        concentrations[j] = sum_sim / count
    else:
        concentrations[j] = 0.0


@cuda.jit
def immune_sweep_kernel(
    rng_states,
    x,
    a,
    D_next,
    M_next,
    payoffs,
    concentrations,
    new_x,
    new_a,
    neighbor_indptr,
    neighbor_indices,
    num_nodes,
    w_payoff,
    w_memory,
    K_P,
    K_M,
    global_M_sum_buf,
    alpha_conc,
    sigma_min,
    sigma_max,
    learning_rate,
):
    """
    MDAE affinity + concentration suppression + adaptive perturbation
    + inertial strategy update.

    Candidate set for individual i: {i} union N_i.
    P affinity uses candidate payoff relative to local candidate average.
    M affinity uses candidate memory relative to global average memory.
    Technical debt remains in debt dynamics, payoff, and reliability-memory
    formation, but is absent from the concentration gate. The gate directly
    combines local concentration with normalized unreliability, rho(1-M').
    """
    i = cuda.grid(1)
    if i >= num_nodes:
        return

    global_avg_M = global_M_sum_buf[0] / num_nodes

    # Local payoff baseline over candidate set {i} union N_i.
    sum_P = payoffs[i]
    candidate_count = 1.0

    start = neighbor_indptr[i]
    end = neighbor_indptr[i + 1]
    for p in range(start, end):
        j = neighbor_indices[p]
        sum_P += payoffs[j]
        candidate_count += 1.0

    local_avg_P = sum_P / candidate_count

    # Candidate: self.
    P_j = payoffs[i]
    M_j = M_next[i]
    rho_j = concentrations[i]

    P_norm = _sigmoid_diff(P_j - local_avg_P, K_P)
    M_norm = _sigmoid_diff(M_j - global_avg_M, K_M)
    affinity = _clip01(w_payoff * P_norm + w_memory * M_norm)
    penalty = alpha_conc * rho_j * (1.0 - M_norm)
    best_stim = affinity * math.exp(-penalty)
    best_node = i
    parent_affinity = affinity

    # Candidates: neighbors.
    for p in range(start, end):
        j = neighbor_indices[p]

        P_j = payoffs[j]
        M_j = M_next[j]
        rho_j = concentrations[j]

        P_norm = _sigmoid_diff(P_j - local_avg_P, K_P)
        M_norm = _sigmoid_diff(M_j - global_avg_M, K_M)
        affinity = _clip01(w_payoff * P_norm + w_memory * M_norm)
        penalty = alpha_conc * rho_j * (1.0 - M_norm)
        stim = affinity * math.exp(-penalty)

        if stim > best_stim:
            best_stim = stim
            best_node = j
            parent_affinity = affinity

    sigma = sigma_min + (sigma_max - sigma_min) * (1.0 - parent_affinity)
    if sigma < 0.0:
        sigma = 0.0

    noise_x = xoroshiro128p_normal_float64(rng_states, i) * sigma
    noise_a = xoroshiro128p_normal_float64(rng_states, i) * sigma

    target_x = _clip01(x[best_node] + noise_x)
    target_a = _clip01(a[best_node] + noise_a)

    updated_x = (1.0 - learning_rate) * x[i] + learning_rate * target_x
    updated_a = (1.0 - learning_rate) * a[i] + learning_rate * target_a

    new_x[i] = _clip01(updated_x)
    new_a[i] = _clip01(updated_a)


# ==============================================================================
# Controller class: Watts--Strogatz small-world network
# ==============================================================================


class AITechnicalDebtMDAE_SmallWorldGPU:
    """
    Small-world network GPU implementation of the AI-induced technical debt
    dual-strategy public goods game.

    Network definition:
        Undirected Watts--Strogatz small-world network.
        Each node starts with ``small_world_k`` ring-lattice neighbours, and
        each clockwise edge is rewired with probability ``small_world_p``.
        Node i has neighborhood N_i.
        Local public goods group centered at g is G_g = {g} union N_g.
        Individual i participates in groups centered at {i} union N_i.

    State variables:
        x: quality assurance investment, [0, 1]
        a: AI use intensity, [0, 1]
        D: AI-induced technical debt, [0, 1]
        M: reliability memory, [0, 1]

    This class only advances model states and reads states. It does not contain
    parameter scans, steady-state aggregation, plotting, or file export logic.
    """

    def __init__(
        self,
        num_nodes: int = 10000,
        seed: Optional[int] = None,
        # Watts--Strogatz small-world network
        small_world_k: int = 4,
        small_world_p: float = 0.1,
        # external AI environment
        A_env: float = 0.5,
        # technical debt dynamics
        delta_D: float = 0.1,
        beta_a: float = 0.5,
        omega_D: float = 0.5,
        beta_N: float = 0.1,
        mu_x: float = 0.3,
        kappa_D: float = 0.1,
        spillover_include_self: bool = True,
        # public goods payoff
        r_public: float = 3.5,
        theta_D: float = 1.0,
        b_ai: float = 1.0,
        chi_x: float = 0.3,
        c_x: float = 1.0,
        c_a: float = 0.1,
        tau_D: float = 1.0,
        # reliability memory
        lambda_0: float = 0.1,
        lambda_A: float = 0.4,
        u_D: float = 2.0,
        # MDAE affinity
        w_payoff: float = 0.7,
        w_memory: float = 0.3,
        K_P: float = 0.2,
        K_M: float = 0.2,
        # reliability-conditioned concentration regulation
        delta: float = 0.3,
        lambda_a: float = 1.0,
        concentration_include_self: bool = False,
        alpha_conc: float = 3.0,
        # adaptive perturbation and inertia
        sigma_min: float = 0.001,
        sigma_max: float = 0.03,
        learning_rate: float = 0.5,
        # optional initial states
        init_x: Optional[np.ndarray] = None,
        init_a: Optional[np.ndarray] = None,
        init_D: Optional[np.ndarray] = None,
        init_M: Optional[np.ndarray] = None,
        init_D_max: float = 0.02,
        # CUDA configuration
        threadsperblock: int = 256,
    ):
        self.num_nodes = int(num_nodes)
        self.seed = int(time.time()) if seed is None else int(seed)
        self.small_world_k = int(small_world_k)
        self.small_world_p = float(small_world_p)

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

        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.learning_rate = float(learning_rate)

        self.init_D_max = float(init_D_max)

        self.threadsperblock = int(threadsperblock)
        if self.threadsperblock <= 0:
            raise ValueError("threadsperblock must be positive.")
        self.blockspergrid = (math.ceil(self.num_nodes / self.threadsperblock),)

        self._validate_parameters()
        self._init_network()
        self._init_device_arrays(init_x=init_x, init_a=init_a, init_D=init_D, init_M=init_M)
        self.iteration = 0

    def _validate_parameters(self) -> None:
        if self.num_nodes <= 0:
            raise ValueError("num_nodes must be positive.")
        if self.num_nodes > 2:
            if self.small_world_k < 2:
                raise ValueError("small_world_k must be at least 2.")
            if self.small_world_k >= self.num_nodes:
                raise ValueError("small_world_k must be smaller than num_nodes.")
            if self.small_world_k % 2 != 0:
                raise ValueError("small_world_k must be even.")
        if not (0.0 <= self.small_world_p <= 1.0):
            raise ValueError("small_world_p must be in [0, 1].")
        if not (0.0 <= self.A_env <= 1.0):
            raise ValueError("A_env must be in [0, 1].")
        if self.kappa_D < 0.0:
            raise ValueError("kappa_D must be non-negative.")
        if self.delta < 0.0:
            raise ValueError("delta must be non-negative.")
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

    @staticmethod
    def _adjacency_to_csr(adjacency):
        indptr = np.empty(len(adjacency) + 1, dtype=np.int32)
        total_edges = 0
        indptr[0] = 0
        for node, neighbors in enumerate(adjacency):
            total_edges += len(neighbors)
            indptr[node + 1] = total_edges

        indices = np.empty(total_edges, dtype=np.int32)
        cursor = 0
        for neighbors in adjacency:
            for neighbor in sorted(neighbors):
                indices[cursor] = int(neighbor)
                cursor += 1

        return indptr, indices

    @staticmethod
    def _build_small_world_csr(num_nodes: int, k: int, p: float, seed: int):
        """Build an undirected Watts--Strogatz small-world network."""
        adjacency = [set() for _ in range(num_nodes)]

        if num_nodes <= 1:
            return AITechnicalDebtMDAE_SmallWorldGPU._adjacency_to_csr(adjacency)

        rng = np.random.default_rng(seed)
        k = max(2, min(int(k), num_nodes - 1))
        if k % 2 != 0:
            k -= 1
        half_k = k // 2

        # Regular ring lattice. Each clockwise edge is created exactly once.
        original_edges = []
        for source in range(num_nodes):
            for offset in range(1, half_k + 1):
                target = (source + offset) % num_nodes
                adjacency[source].add(target)
                adjacency[target].add(source)
                original_edges.append((source, target))

        # Watts--Strogatz rewiring preserves the edge count and the source end.
        for source, old_target in original_edges:
            if rng.random() >= p:
                continue
            if old_target not in adjacency[source]:
                continue

            adjacency[source].remove(old_target)
            adjacency[old_target].remove(source)

            candidates_left = num_nodes - 1 - len(adjacency[source])
            if candidates_left <= 0:
                adjacency[source].add(old_target)
                adjacency[old_target].add(source)
                continue

            new_target = int(rng.integers(num_nodes))
            while new_target == source or new_target in adjacency[source]:
                new_target = int(rng.integers(num_nodes))
            adjacency[source].add(new_target)
            adjacency[new_target].add(source)

        return AITechnicalDebtMDAE_SmallWorldGPU._adjacency_to_csr(adjacency)

    def _init_network(self) -> None:
        indptr, indices = self._build_small_world_csr(
            num_nodes=self.num_nodes,
            k=self.small_world_k,
            p=self.small_world_p,
            seed=self.seed,
        )

        self.neighbor_indptr_host = np.ascontiguousarray(indptr)
        self.neighbor_indices_host = np.ascontiguousarray(indices)
        self.degree_host = np.diff(self.neighbor_indptr_host).astype(np.int32)
        self.num_edges = int(len(self.neighbor_indices_host) // 2)

        self.d_neighbor_indptr = cuda.to_device(self.neighbor_indptr_host)
        self.d_neighbor_indices = cuda.to_device(self.neighbor_indices_host)

    def get_network_info(self) -> Dict[str, float]:
        degrees = self.degree_host
        if len(degrees) == 0:
            mean_degree = 0.0
            max_degree = 0
            min_degree = 0
        else:
            mean_degree = float(np.mean(degrees))
            max_degree = int(np.max(degrees))
            min_degree = int(np.min(degrees))

        return {
            "network_type": "small_world",
            "small_world_k": int(self.small_world_k),
            "small_world_p": float(self.small_world_p),
            "nodes": int(self.num_nodes),
            "edges": int(self.num_edges),
            "mean_degree": mean_degree,
            "min_degree": min_degree,
            "max_degree": max_degree,
        }

    def _random_array_with_exact_mean_half(self, rng: np.random.Generator) -> np.ndarray:
        """
        Generate a random array in [0, 1] with exact mean 0.5.
        """
        total_size = self.num_nodes
        half_size = total_size // 2

        first_half = rng.random(half_size, dtype=np.float64)
        values = np.empty(total_size, dtype=np.float64)
        values[:half_size] = first_half
        values[half_size:2 * half_size] = 1.0 - first_half

        if total_size % 2 == 1:
            values[-1] = 0.5

        rng.shuffle(values)
        return np.ascontiguousarray(values)

    def _prepare_init_array(
        self,
        arr: Optional[np.ndarray],
        rng: np.random.Generator,
        fill: Optional[float] = None,
        exact_mean_half: bool = False,
        low_debt: bool = False,
    ) -> np.ndarray:
        if arr is None:
            if exact_mean_half:
                out = self._random_array_with_exact_mean_half(rng)
            elif low_debt:
                if self.init_D_max == 0.0:
                    out = np.zeros(self.num_nodes, dtype=np.float64)
                else:
                    out = rng.random(self.num_nodes, dtype=np.float64) * self.init_D_max
            elif fill is not None:
                out = np.full(self.num_nodes, fill, dtype=np.float64)
            else:
                out = rng.random(self.num_nodes, dtype=np.float64)
        else:
            out = np.asarray(arr, dtype=np.float64)
            if out.shape != (self.num_nodes,):
                raise ValueError(f"Initial array shape must be ({self.num_nodes},), got {out.shape}.")
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

        x_host = self._prepare_init_array(init_x, rng, exact_mean_half=True)
        a_host = self._prepare_init_array(init_a, rng, exact_mean_half=True)
        D_host = self._prepare_init_array(init_D, rng, low_debt=True)
        M_host = self._prepare_init_array(init_M, rng, fill=0.5)

        self.d_x = cuda.to_device(x_host)
        self.d_a = cuda.to_device(a_host)
        self.d_D = cuda.to_device(D_host)
        self.d_M = cuda.to_device(M_host)

        self.d_D_next = cuda.device_array(self.num_nodes, dtype=np.float64)
        self.d_M_next = cuda.device_array(self.num_nodes, dtype=np.float64)
        self.d_payoff = cuda.device_array(self.num_nodes, dtype=np.float64)
        self.d_concentration = cuda.device_array(self.num_nodes, dtype=np.float64)
        self.d_group_benefit = cuda.device_array(self.num_nodes, dtype=np.float64)
        self.d_new_x = cuda.device_array(self.num_nodes, dtype=np.float64)
        self.d_new_a = cuda.device_array(self.num_nodes, dtype=np.float64)
        self.d_global_M_sum = cuda.device_array(1, dtype=np.float64)

        self.rng_states = create_xoroshiro128p_states(self.num_nodes, seed=self.seed)

    def set_ai_environment(self, A_env: float) -> None:
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
        rebuild_network: bool = True,
    ) -> None:
        """
        Reset model state. If seed changes and rebuild_network=True, the
        small-world network is rebuilt with the new seed.
        """
        if seed is not None:
            self.seed = int(seed)
        if init_D_max is not None:
            self.init_D_max = float(init_D_max)

        self._validate_parameters()
        if rebuild_network:
            self._init_network()
        self._init_device_arrays(init_x=init_x, init_a=init_a, init_D=init_D, init_M=init_M)
        self.iteration = 0

    def step(self, synchronize: bool = False) -> None:
        """
        Advance one synchronous evolutionary step.
        """
        lambda_M = self._lambda_M()

        update_debt_kernel[self.blockspergrid, self.threadsperblock](
            self.d_x,
            self.d_a,
            self.d_D,
            self.d_D_next,
            self.d_neighbor_indptr,
            self.d_neighbor_indices,
            self.num_nodes,
            self.delta_D,
            self.beta_a,
            self.omega_D,
            self.beta_N,
            self.mu_x,
            self.kappa_D,
            self.spillover_include_self,
        )

        calc_group_benefit_kernel[self.blockspergrid, self.threadsperblock](
            self.d_x,
            self.d_D_next,
            self.d_group_benefit,
            self.d_neighbor_indptr,
            self.d_neighbor_indices,
            self.num_nodes,
            self.r_public,
            self.theta_D,
        )

        calc_payoffs_from_group_benefit_kernel[self.blockspergrid, self.threadsperblock](
            self.d_x,
            self.d_a,
            self.d_D_next,
            self.d_group_benefit,
            self.d_payoff,
            self.d_neighbor_indptr,
            self.d_neighbor_indices,
            self.num_nodes,
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
            self.num_nodes,
            lambda_M,
            self.u_D,
        )

        zero_scalar_kernel[1, 1](self.d_global_M_sum)
        reduce_sum_kernel[self.blockspergrid, self.threadsperblock](
            self.d_M_next,
            self.d_global_M_sum,
            self.num_nodes,
        )

        calc_concentrations_kernel[self.blockspergrid, self.threadsperblock](
            self.d_x,
            self.d_a,
            self.d_concentration,
            self.d_neighbor_indptr,
            self.d_neighbor_indices,
            self.num_nodes,
            self.delta,
            self.lambda_a,
            self.concentration_include_self,
        )

        immune_sweep_kernel[self.blockspergrid, self.threadsperblock](
            self.rng_states,
            self.d_x,
            self.d_a,
            self.d_D_next,
            self.d_M_next,
            self.d_payoff,
            self.d_concentration,
            self.d_new_x,
            self.d_new_a,
            self.d_neighbor_indptr,
            self.d_neighbor_indices,
            self.num_nodes,
            self.w_payoff,
            self.w_memory,
            self.K_P,
            self.K_M,
            self.d_global_M_sum,
            self.alpha_conc,
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
        for _ in range(int(steps)):
            self.step(synchronize=False)
        if synchronize:
            cuda.synchronize()

    def get_state(self) -> Dict[str, np.ndarray]:
        return {
            "x": self.d_x.copy_to_host(),
            "a": self.d_a.copy_to_host(),
            "D": self.d_D.copy_to_host(),
            "M": self.d_M.copy_to_host(),
        }

    def get_observables(self) -> Dict[str, np.ndarray]:
        return {
            "payoff": self.d_payoff.copy_to_host(),
            "concentration": self.d_concentration.copy_to_host(),
            "group_benefit": self.d_group_benefit.copy_to_host(),
        }

    def get_means(self) -> Dict[str, float]:
        state = self.get_state()
        return {
            "mean_x": float(np.mean(state["x"])),
            "mean_a": float(np.mean(state["a"])),
            "mean_D": float(np.mean(state["D"])),
            "mean_M": float(np.mean(state["M"])),
        }

    def device_arrays(self) -> Dict[str, object]:
        return {
            "x": self.d_x,
            "a": self.d_a,
            "D": self.d_D,
            "M": self.d_M,
            "payoff": self.d_payoff,
            "concentration": self.d_concentration,
            "group_benefit": self.d_group_benefit,
        }
