# ---------------------------------------------------------
#  Random graph generator for Γ_{μ_min,b} with rich comments
# ---------------------------------------------------------
#
#  Γ_{μ_min,b} = { G :  algebraic connectivity μ2(G_H) ≥ μ_min
#                         and   max_{i∈H}  b(i) ≤ b }
#
#  - G_H : subgraph induced by honest nodes
#  - b(i): number of Byzantine neighbours of honest node i (unit weights)
#
#  We build graphs in two stages:
#    1. Generate a k-regular honest sub-graph with μ2 ≥ μ_min
#    2. Attach Byzantine nodes, making sure each honest node
#       has at most f ≤ b Byzantine neighbours
#
#  Libraries needed: networkx, numpy, pandas, matplotlib
# ---------------------------------------------------------

import networkx as nx 
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import eigvalsh
from typing import Iterable, Dict, Tuple, List
import sys
# from ace_tools import display_dataframe_to_user

# Flag to control whether to show plots
# Only show plots when run as main script, not when imported
SHOW_PLOTS = __name__ == '__main__'

# -----------------------------------------------------------------------------
#  Laplacian construction (unit weights)
# -----------------------------------------------------------------------------

def laplacian_matrix_unit(G: nx.Graph) -> np.ndarray:
    """Return the *unit-weight* Laplacian matrix as defined in the paper.

    Parameters
    ----------
    G : nx.Graph
        Undirected graph (edge attribute *weights* ignored, unit weight assumed).

    Returns
    -------
    L : ndarray  shape (n,n)
        The combinatorial Laplacian defined by   L = D - A   with unit weights.
    """
    n = G.number_of_nodes()
    # consistent node ordering
    nodes: List = list(G.nodes())
    idx: Dict = {v: i for i, v in enumerate(nodes)}

    L = np.zeros((n, n), dtype=float)
    # fill off-diagonals with -1 for each edge
    for u, v in G.edges():
        i, j = idx[u], idx[v]
        L[i, j] -= 1.0
        L[j, i] -= 1.0
        L[i, i] += 1.0   # degree contribution for u
        L[j, j] += 1.0   # degree contribution for v
    return L

# ---------- utility: algebraic connectivity ---------------------------------
def algebraic_connectivity(G):
    """
    Return μ₂, the second-smallest eigenvalue of the (unweighted) Laplacian.
    For connected graphs μ₂ > 0 ; larger ⇒ better connectivity.
    """

    # evals = eigvalsh(nx.laplacian_matrix(G).astype(float).todense())
    # print(f"Graph: \n{laplacian_matrix_unit(G)}")
    evals = eigvalsh(laplacian_matrix_unit(G).astype(float))
    # print(f"Eigenvalues: {sorted(evals)}")
    mu2 = sorted(evals)[1]
    mu2 = max(mu2, 0.0)
    return mu2


# ---------- step-1: honest sub-graph ----------------------------------------
def honest_regular_graph(n_honest: int,
                         k: int,
                         mu_min: float,
                         seed=None,
                         max_trials: int = 500):
    """
    Randomly sample k-regular graphs until algebraic connectivity ≥ mu_min.

    Parameters
    ----------
    n_honest : int
        number of honest nodes (graph order)
    k : int
        desired regular degree  (must satisfy 0 ≤ k < n_honest and n_honest*k even)

    mu_min : float
        lower bound we want on μ₂(G_H)
    seed : int, optional
        PRNG seed for reproducibility
    max_trials : int
        safety cap for rejection sampling

    Returns
    -------
    G : networkx.Graph
        the k-regular honest graph that meets μ₂ ≥ mu_min
    μ2 : float
        its algebraic connectivity
    """
    rnd = random.Random(seed)

    for _ in range(max_trials):
        # networkx.random_regular_graph needs "n * k" even
        G = nx.random_regular_graph(k, n_honest, seed=rnd.randint(0, 2**32 - 1))
        mu2 = algebraic_connectivity(G)
        if mu2 >= mu_min:
            return G, mu2

    raise RuntimeError(f"failed to reach μ₂ ≥ {mu_min} within {max_trials} trials")


def honest_er_graph(n_honest: int,
                    p: float,
                    mu_min: float,
                    seed=None,
                    max_trials: int = 500):
    """
    Randomly sample G(n_honest, p) Erdős–Rényi graphs until μ₂(G) ≥ mu_min.
    Edge weights are unitary. Returns (G, μ₂).
    """
    rnd = np.random.default_rng(seed)
    for _ in range(max_trials):
        G = nx.erdos_renyi_graph(n_honest, p, seed=int(rnd.integers(1e9)))
        # ensure graph is connected
        if not nx.is_connected(G):
            continue
        mu2 = algebraic_connectivity(G)
        if mu2 >= mu_min:
            return G, mu2
    raise RuntimeError(f"failed to reach μ₂ ≥ {mu_min} with ER(n={n_honest}, p={p})")


def honest_ws_graph(n_honest: int,
                    k: int,
                    p: float,
                    mu_min: float,
                    seed=None,
                    max_trials: int = 500):
    """
    Randomly sample G(n_honest) Watts-Strogatz small-world graphs until μ₂(G) ≥ mu_min.
    Unweighted edges, returns (G, μ₂).
    """
    rnd = np.random.default_rng(seed)
    for _ in range(max_trials):
        G = nx.watts_strogatz_graph(n_honest, k, p, seed=int(rnd.integers(1e9)))
        if not nx.is_connected(G):
            continue
        mu2 = algebraic_connectivity(G)
        if mu2 >= mu_min:
            return G, mu2
    raise RuntimeError(f"failed to reach μ₂ ≥ {mu_min} with WS(n={n_honest}, k={k}, p={p})")


def honest_ba_graph(n_honest: int,
                    m: int,
                    mu_min: float,
                    seed=None,
                    max_trials: int = 500):
    """
    Randomly sample Barabasi-Albert graphs until μ₂(G) ≥ mu_min.
    Returns (G, μ₂).
    """
    rnd = np.random.default_rng(seed)
    for _ in range(max_trials):
        G = nx.barabasi_albert_graph(n_honest, m, seed=int(rnd.integers(1e9)))
        if not nx.is_connected(G):
            continue
        mu2 = algebraic_connectivity(G)
        if mu2 >= mu_min:
            return G, mu2
    raise RuntimeError(f"failed to reach μ₂ ≥ {mu_min} with BA(n={n_honest}, m={m})")


# ---------- step-2: attach Byzantine nodes ----------------------------------
def attach_byzantines(G_honest: nx.Graph,
                      n_byz: int,
                      f: int,
                      seed=None):
    """
    Add Byzantine nodes and connect each honest node to ≤ f Byzantines.

    Returns the augmented graph.
    Honest nodes keep their original integer labels: 0..n_honest-1
    Byzantine nodes get labels 'B0', 'B1', ...

    Parameters
    ----------
    G_honest : Graph
    n_byz : int
        how many Byzantine nodes to add
    f : int
        per-honest-node budget of Byzantine neighbours ( ≤ b in theory )
    """
    rnd = random.Random(seed)
    G = G_honest.copy()
    honest_nodes = list(G.nodes())

    # add Byzantine nodes
    byz_nodes = [f"B{j}" for j in range(n_byz)]
    for b in byz_nodes:
        G.add_node(b, byzantine=True)

    # track remaining quota for each honest node
    quota = {h: f for h in honest_nodes}

    # connect Byzantines one by one, respecting quota
    for b in byz_nodes:
        candidates = [h for h in honest_nodes if quota[h] > 0]
        # ensure each Byzantine node attaches to at least 1 honest neighbour
        deg_b = rnd.randint(1, min(len(candidates), f))
        neigh = rnd.sample(candidates, deg_b)
        for h in neigh:
            G.add_edge(b, h)
            quota[h] -= 1

    return G


# ---------- convenience wrapper ---------------------------------------------
def gen_example(name: str,
                n: int,
                n_byz: int,
                mu_min: float,
                f: int,
                k: int,
                seed=None):
    """
    Produce one example graph in Γ_{μ_min,b} and collect its statistics.
    """
    n_hon = n - n_byz
    G_h, mu2 = honest_regular_graph(n_hon, k, mu_min, seed)
    # compute densities: honest-only and full graph
    density_honest = round(nx.density(G_h), 3)
    G = attach_byzantines(G_h, n_byz, f, seed)
    density_full = round(nx.density(G), 3)

    byz_set = {v for v in G if isinstance(v, str) and v.startswith("B")}
    max_b = max(sum(1 for nb in G.neighbors(h) if nb in byz_set) for h in G_h)

    stats = {"example": name,
             "n_hon": n_hon, "n_byz": n_byz,
             "k": k, "f": f,
             "μ₂(G_H)": round(mu2, 3),
             "max_b(i)": max_b,
             "density_honest": density_honest,
             "density_full": density_full}

    mu2 = stats['μ₂(G_H)']
    if 2 * stats['f'] >= mu2:
        print(f"Warning: 2*b = {2*stats['f']} >= μ₂ = {mu2:.3f} for example '{name}'")

    return G, stats


def gen_example_er(name: str,
                   n: int,
                   n_byz: int,
                   p: float,
                   mu_min: float,
                   f: int,
                   seed=None):
    """
    Generate an ER(n_honest, p) honest subgraph with μ₂ ≥ mu_min, attach Byzantines, return (G, stats).
    """
    n_hon = n - n_byz
    G_h, mu2 = honest_er_graph(n_hon, p, mu_min, seed)
    # compute densities: honest-only and full graph
    density_honest = round(nx.density(G_h), 3)
    G = attach_byzantines(G_h, n_byz, f, seed)
    density_full = round(nx.density(G), 3)

    byz_set = {v for v in G if isinstance(v, str) and v.startswith("B")}
    max_b = max(sum(1 for nb in G.neighbors(h) if nb in byz_set) for h in range(n_hon))
    stats = {"example": name,
             "n_hon": n_hon,
             "n_byz": n_byz,
             "p": p,
             "mu_min": mu_min,
             "f": f,
             "mu2": round(mu2, 3),
             "max_b": max_b,
             "density_honest": density_honest,
             "density_full": density_full}

    mu2 = stats['mu2']
    if 2 * stats['f'] >= mu2:
        print(f"Warning: 2*b = {2*stats['f']} >= μ₂ = {mu2:.3f} for example '{name}'")

    return G, stats


def gen_example_ws(name: str,
                   n: int,
                   n_byz: int,
                   k: int,
                   p: float,
                   mu_min: float,
                   f: int,
                   seed=None):
    """
    Generate a Watts-Strogatz honest subgraph with μ₂ ≥ mu_min, attach Byzantines.
    Returns (G, stats).
    """
    n_hon = n - n_byz
    G_h, mu2 = honest_ws_graph(n_hon, k, p, mu_min, seed)
    # compute densities: honest-only and full graph
    density_honest = round(nx.density(G_h), 3)
    G = attach_byzantines(G_h, n_byz, f, seed)
    density_full = round(nx.density(G), 3)

    byz_set = {v for v in G if isinstance(v, str) and v.startswith("B")}
    max_b = max(sum(1 for nb in G.neighbors(h) if nb in byz_set) for h in range(n_hon))
    stats = {"example": name,
             "n_hon": n_hon,
             "n_byz": n_byz,
             "type": "watts-strogatz",
             "k": k,
             "p": p,
             "mu_min": mu_min,
             "f": f,
             "mu2": round(mu2, 3),
             "max_b": max_b,
             "density_honest": density_honest,
             "density_full": density_full}
    return G, stats


def gen_example_ba(name: str,
                   n: int,
                   n_byz: int,
                   m: int,
                   mu_min: float,
                   f: int,
                   seed=None):
    """
    Generate a Barabási-Albert honest subgraph with μ₂ ≥ mu_min, attach Byzantines.
    Returns (G, stats).
    """
    n_hon = n - n_byz
    G_h, mu2 = honest_ba_graph(n_hon, m, mu_min, seed)
    # compute densities: honest-only and full graph
    density_honest = round(nx.density(G_h), 3)
    G = attach_byzantines(G_h, n_byz, f, seed)
    density_full = round(nx.density(G), 3)

    byz_set = {v for v in G if isinstance(v, str) and v.startswith("B")}
    max_b = max(sum(1 for nb in G.neighbors(h) if nb in byz_set) for h in range(n_hon))
    stats = {"example": name,
             "n_hon": n_hon,
             "n_byz": n_byz,
             "type": "barabasi-albert",
             "m": m,
             "mu_min": mu_min,
             "f": f,
             "mu2": round(mu2, 3),
             "max_b": max_b,
             "density_honest": density_honest,
             "density_full": density_full}
    return G, stats


# ---------- generate several examples ---------------------------------------
# f is the per-honest-node budget of Byzantine neighbours since we assume unit weights f = b_max 
#   where b_max is the maximum number of Byzantine nodes of any honest node
# k is the degree of the honest subgraph (k-regular)
# n is the total number of nodes (n_honest + n_byz)
# n_byz is the number of Byzantine nodes
# mu_min is the lower bound we want on μ₂(G_H)  

examples_params = [
    # You can edit / add rows to explore other configurations
    dict(name="Expander-k6",        n=20, n_byz=5, mu_min=2,   f=3, k=6, seed=1),
    dict(name="Denser-k8",          n=20, n_byz=5, mu_min=4,   f=2, k=8, seed=2),
    dict(name="Clique-k14",         n=20, n_byz=5, mu_min=10,  f=2, k=14, seed=3),
    dict(name="Sparse-k4",          n=20, n_byz=5, mu_min=1.5, f=1, k=4, seed=4),
    dict(name="Erdős-Rényi feel",   n=20, n_byz=5, mu_min=1.2, f=2, k=6, seed=5),
    dict(name="VerySparse-k2",  n=20, n_byz=5, mu_min=0.05, f=1, k=2, seed=10),
    dict(name="MediumSparse-k3", n=22, n_byz=6, mu_min=1.0, f=1, k=3, seed=11),
]

summary_rows = []
for p in examples_params:
    G, stats = gen_example(**p)
    summary_rows.append(stats)
    # skip plotting if 2*b >= μ₂
    if 2 * stats['f'] >= stats['μ₂(G_H)']:
        print(f"Warning: 2*b = {2*stats['f']} >= μ₂ = {stats['μ₂(G_H)']:.3f} for example '{p['name']}'")
        continue

    # ----------- one figure per graph (guideline: no subplots) --------------
    if SHOW_PLOTS:
        byz_nodes = {v for v in G if isinstance(v, str) and v.startswith("B")}
        node_colors = ["tab:red" if v in byz_nodes else "tab:blue" for v in G.nodes()]

        plt.figure(figsize=(4, 4))
        pos = nx.spring_layout(G, seed=123)          # deterministic layout for clarity
        nx.draw_networkx(G, pos, node_color=node_colors, node_size=180,
                         with_labels=False, edge_color="gray")
        density = nx.density(G)
        plt.title(f"{p['name']}\nμ₂={stats['μ₂(G_H)']},  max b={stats['max_b(i)']}, density={density:.3f}")
        plt.axis("off")
        plt.show()

# ---------- generate ER examples ---------------------------------------
er_params = [
    dict(name="ER-p0.2", n=20, n_byz=5, p=0.2, mu_min=1.0, f=2, seed=42),
    dict(name="ER-p0.5", n=20, n_byz=5, p=0.5, mu_min=1.0, f=2, seed=24),
]
er_rows = []
for p in er_params:
    G, stats = gen_example_er(**p)
    er_rows.append(stats)
    # skip plotting if 2*b >= μ₂
    if 2 * stats['f'] >= stats['mu2']:
        continue

    if SHOW_PLOTS:
        byz_nodes = {v for v in G if isinstance(v, str) and v.startswith("B")}
        node_colors = ["tab:red" if v in byz_nodes else "tab:blue" for v in G.nodes()]
        plt.figure(figsize=(4,4))
        nx.draw_networkx(G, pos=nx.spring_layout(G, seed=123),
                         node_color=node_colors, node_size=160,
                         edge_color="gray", with_labels=False)
        density = nx.density(G)
        plt.title(f"{p['name']}\nμ₂={stats['mu2']},  b={stats['f']}, density={density:.3f}")
        plt.axis("off")
        plt.show()

df_er = pd.DataFrame(er_rows)
print("\nER examples summary:")
print(df_er)

# ---------- generate WS examples ---------------------------------------
ws_params = [
    dict(name="WS-k4-p0.1", n=20, n_byz=5, k=4, p=0.1, mu_min=1.0, f=2, seed=42),
    dict(name="WS-k6-p0.3", n=20, n_byz=5, k=6, p=0.3, mu_min=1.0, f=2, seed=24),
]
ws_rows = []
for p in ws_params:
    G, stats = gen_example_ws(**p)
    ws_rows.append(stats)
    # skip plotting if 2*b >= μ₂
    if 2 * stats['f'] >= stats['mu2']:
        continue

    if SHOW_PLOTS:
        byz_nodes = {v for v in G if isinstance(v, str) and v.startswith("B")}
        node_colors = ["tab:red" if v in byz_nodes else "tab:blue" for v in G.nodes()]
        plt.figure(figsize=(4,4))
        nx.draw_networkx(G, pos=nx.spring_layout(G, seed=123),
                         node_color=node_colors, node_size=160,
                         edge_color="gray", with_labels=False)
        density = nx.density(G)
        plt.title(f"{p['name']}\nμ₂={stats['mu2']},  b={stats['f']}, density={density:.3f}")
        plt.axis("off")
        plt.show()

df_ws = pd.DataFrame(ws_rows)
print("\nWS examples summary:")
print(df_ws)

# ---------- generate BA examples ---------------------------------------
ba_params = [
    dict(name="BA-m2", n=20, n_byz=5, m=2, mu_min=1.0, f=2, seed=42),
    dict(name="BA-m3", n=20, n_byz=5, m=3, mu_min=1.0, f=2, seed=24),
]
ba_rows = []
for p in ba_params:
    G, stats = gen_example_ba(**p)
    ba_rows.append(stats)
    # skip plotting if 2*b >= μ₂
    if 2 * stats['f'] >= stats['mu2']:
        continue

    if SHOW_PLOTS:
        byz_nodes = {v for v in G if isinstance(v, str) and v.startswith("B")}
        node_colors = ["tab:red" if v in byz_nodes else "tab:blue" for v in G.nodes()]
        plt.figure(figsize=(4,4))
        nx.draw_networkx(G, pos=nx.spring_layout(G, seed=123),
                         node_color=node_colors, node_size=160,
                         edge_color="gray", with_labels=False)
        density = nx.density(G)
        plt.title(f"{p['name']}\nμ₂={stats['mu2']},  b={stats['f']}, density={density:.3f}")
        plt.axis("off")
        plt.show()

df_ba = pd.DataFrame(ba_rows)
print("\nBA examples summary:")
print(df_ba)

# ---------- nice table of statistics ----------------------------------------
df_summary = pd.DataFrame(summary_rows)
# display_dataframe_to_user("Graph examples in Γ_{μ_min,b}", df_summary)
