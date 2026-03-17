import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO

# FCI
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.cit import CIT
from causallearn.utils.GraphUtils import GraphUtils
import io
from PIL import Image

import pydot

fixed_pos = {

    # t-1 (izquierda)
    "bal_150_t-1": (40, 157.56),
    "bal_250_t-1": (80, 200.64),
    "bal_energy_band_low_t-1": (50, 83.773),
    "bal_energy_band_medium_t-1": (85, 72.708),
    "bal_energy_band_high_t-1": (75, 125.12),
    "6004_FTF_t-1": (100, 131.84),
    "bal_350_t-1": (120, 91.955),
    "6004_BPFI_t-1": (30, 18.0),
    "6004_BSF_t-1": (95, 51.218),
    "6004_BPFO_t-1": (15, 84.6),

    # t (derecha)
    "bal_150_t": (240, 157.56),
    "bal_250_t": (280, 200.64),
    "bal_energy_band_low_t": (250, 83.773),
    "bal_energy_band_medium_t": (285, 72.708),
    "bal_energy_band_high_t": (275, 125.12),
    "6004_FTF_t": (300, 131.84),
    "bal_350_t": (320, 91.955),
    "6004_BPFI_t": (230, 18.0),
    "6004_BSF_t": (295, 51.218),
    "6004_BPFO_t": (215, 84.6),
}

# LiNGAM (causal-learn)
from causallearn.search.FCMBased import lingam
from matplotlib.colors import SymLogNorm
# For drawing LiNGAM as a graph
import networkx as nx

CSV_PATH = r".\anomalous.csv" # CAMBIAR por normal.csv para ver datos normales 

SEP = ";"
TIME_COL = "timestamp"
TIMESTAMP_UNIT = "us"  

GAP = pd.Timedelta(days=1)     # split whenever gap > 1 day
COLUMNS_TO_PLOT = None         # None = all columns except timestamp
DOWNSAMPLE_EVERY_N = 1         # set to 5/10 if slow

def show_pydot_graph(pydot_graph, title=""):
    """Render a pydot graph to an image and display with matplotlib."""
    png_bytes = pydot_graph.create_png()
    img = plt.imread(BytesIO(png_bytes))
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.axis("off")
    if title:
        plt.title(title)
    plt.show()




def _overlay_grey_diagonal(n):
    diag_mask = np.zeros((n, n), dtype=float)
    np.fill_diagonal(diag_mask, 1.0)
    plt.imshow(diag_mask, cmap="Greys", alpha=diag_mask)  # full grey squares on diagonal


def plot_lingam_matrix(B, labels, title, percentile=100, eps=1e-12):
    lim = np.percentile(np.abs(B), percentile)
    lim = max(lim, eps)

    n = len(labels)
    plt.figure(figsize=(7, 6))

    # BLUE = positive, RED = negative
    im = plt.imshow(B, vmin=-lim, vmax=lim, cmap="seismic_r")

    # paint diagonal with Greys (overlay)
    _overlay_grey_diagonal(n)

    plt.xticks(range(n), labels, rotation=90)
    plt.yticks(range(n), labels)
    plt.title(f"{title} (±{lim:.3g}, p{percentile})")

    # IMPORTANT: colorbar must be linked to the LiNGAM image, not the diagonal overlay
    plt.colorbar(im)

    plt.tight_layout()
    plt.show()


def plot_lingam_matrix_symlog(B, labels, title, linthresh=0.5):
    vmax = np.max(np.abs(B))
    vmax = max(vmax, 1e-12)

    norm = SymLogNorm(linthresh=linthresh, vmin=-vmax, vmax=vmax, base=10)

    n = len(labels)
    plt.figure(figsize=(7, 6))

    # BLUE = positive, RED = negative
    im = plt.imshow(B, cmap="seismic_r", norm=norm)

    # paint diagonal with Greys (overlay)
    _overlay_grey_diagonal(n)

    plt.xticks(range(n), labels, rotation=90)
    plt.yticks(range(n), labels)
    plt.title(f"{title} (symlog, linthresh={linthresh})")

    plt.colorbar(im)

    plt.tight_layout()
    plt.show()

def plot_lingam_graph(B, labels, threshold=0.0, title="DirectLiNGAM graph"):
    """
    Draw a directed graph from LiNGAM adjacency matrix B.
    threshold: keep edges with |B[i,j]| > threshold
    Interprets B[i, j] as edge j -> i (common convention in LiNGAM).
    """
    G = nx.DiGraph()
    for name in labels:
        G.add_node(name)

    p = B.shape[0]
    for i in range(p):
        for j in range(p):
            w = B[i, j]
            if i != j and abs(w) > threshold:
                # B[i,j] means j -> i
                G.add_edge(labels[j], labels[i], weight=float(w))

    plt.figure(figsize=(9, 7))
    pos = nx.spring_layout(G, seed=0)
    nx.draw_networkx_nodes(G, pos, node_size=1200)
    nx.draw_networkx_labels(G, pos, font_size=9)
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="-|>", width=1.2)
    # edge labels = weights
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def pag_to_endpoint_matrix(g, labels):
    """
    Build a matrix that encodes FCI PAG endpoints.

    For each ordered pair (i, j), matrix[i, j] encodes the endpoint at node i
    on the edge between i and j:

      0 = no edge
      1 = TAIL at i     (i --- *)
      2 = ARROW at i    (i <-  *)
      3 = CIRCLE at i   (i o-- *)

    This gives you a “matrix view” of the PAG (not a simple adjacency).
    """
    # Import here to avoid version issues if unused
    from causallearn.graph.Endpoint import Endpoint

    p = len(labels)
    M = np.zeros((p, p), dtype=int)

    nodes = g.get_nodes()  # order should match data columns in fci()
    # Make sure node count matches labels
    if len(nodes) != p:
        raise ValueError(f"Graph has {len(nodes)} nodes but labels has {p} columns.")

    # Build a quick index: node -> idx
    idx = {nodes[i]: i for i in range(p)}

    for edge in g.get_graph_edges():
        a = edge.get_node1()
        b = edge.get_node2()
        ia, ib = idx[a], idx[b]

        ea = edge.get_endpoint1()  # endpoint at node1
        eb = edge.get_endpoint2()  # endpoint at node2

        def enc(e):
            if e == Endpoint.TAIL:
                return 1
            if e == Endpoint.ARROW:
                return 2
            if e == Endpoint.CIRCLE:
                return 3
            return 0

        M[ia, ib] = enc(ea)
        M[ib, ia] = enc(eb)

    return M


def run_and_plot_on_chunk(df_chunk: pd.DataFrame,
                          alpha=0.05,
                          lingam_threshold=0.0,
                          max_path_length=3,
                          depth=-1):
    # Drop timestamp-like columns if you want (edit list as needed)
    df_chunk = df_chunk.drop(columns=["timestamp", "time", "date"], errors="ignore")

    # Keep only numeric + drop rows with NaNs (algorithms typically require complete cases)
    df_num = df_chunk.select_dtypes(include=[np.number]).dropna(axis=0)
    labels = df_num.columns.to_list()

    if df_num.shape[0] < 10 or df_num.shape[1] < 2:
        print("Chunk skipped (not enough numeric data after cleaning).")
        return

    data = df_num.to_numpy()

    # -------------------
    # 1) DirectLiNGAM: adjacency + graph
    # -------------------
    model = lingam.DirectLiNGAM()
    model.fit(data)
    B = model.adjacency_matrix_

    plot_lingam_matrix(B, labels, "DirectLiNGAM adjacency matrix (B)")
    plot_lingam_matrix_symlog(B, labels, "DirectLiNGAM adjacency matrix (B)")
    plot_lingam_graph(B, labels, threshold=lingam_threshold,
                      title=f"DirectLiNGAM graph (|B| > {lingam_threshold})")

    
    # -------------------
    # 2) FCI: graph + endpoint matrix
    # -------------------
    indep_test = "fisherz"
    g, edges = fci(
        data,
        indep_test,
        alpha=alpha,
        depth=depth,
        max_path_length=max_path_length,
        verbose=False
    )
    
    # Graph drawing (PAG)
    pydot_graph = GraphUtils.to_pydot(g, labels=labels)
    pydot_graph.set_layout("neato")
    
    # Keep output under control
    pydot_graph.set_overlap("false")
    pydot_graph.set_splines("true")
    pydot_graph.set_margin("0.2")
    pydot_graph.set_pad("0.2")
    pydot_graph.set_dpi("100")
    pydot_graph.set_size('"9,7!"')
    
    # Assign fixed positions
    scale = 0.05   # adjust until it looks good
    
    for node in pydot_graph.get_nodes():
        name = node.get_name().strip('"')
    
        if name in {"graph", "node", "edge"}:
            continue
    
        label = node.get_label()
        key = label.strip('"') if label is not None else name
    
        if key in fixed_pos:
            x, y = fixed_pos[key]
            node.set_pos(f"{x * scale},{y * scale}!")
            node.set_pin("true")
    
    # Render with neato
    png_bytes = pydot_graph.create_png(prog="neato")
    
    # Use PIL instead of plt.imread
    img = Image.open(io.BytesIO(png_bytes))
    
    plt.figure(figsize=(9, 7))
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"FCI PAG (alpha={alpha})")
    plt.tight_layout()
    plt.show()
    
    # “Adjacency matrix” for FCI (endpoint-encoding)
    M = pag_to_endpoint_matrix(g, labels)

    n = len(labels)

    plt.figure(figsize=(7,6))
    
    # draw the real matrix and keep the handle
    im = plt.imshow(M, vmin=0, vmax=3)
    
    # diagonal mask
    diag_mask = np.zeros_like(M, dtype=float)
    np.fill_diagonal(diag_mask, 1)
    
    # overlay red diagonal
    plt.imshow(
        diag_mask,
        cmap="Greys",
        alpha=diag_mask
    )
    
    plt.xticks(range(n), labels, rotation=90)
    plt.yticks(range(n), labels)
    
    plt.title("FCI endpoint matrix (0 none, 1 tail, 2 arrow, 3 circle)")
    
    # colorbar linked to the real matrix
    plt.colorbar(im)
    
    plt.tight_layout()
    plt.show()
# -------------------
# Your chunk loop
# -------------------
df = pd.read_csv(CSV_PATH, sep=SEP)
#df = df.drop(columns=["timestamp"])


chunk_size = 10000
prev_chunk = None

for start in range(0, len(df), chunk_size):
    df_chunk = df.iloc[start:start + chunk_size]
    print(f"\n=== Chunk rows {start}..{min(start+chunk_size, len(df))} ===")

    if prev_chunk is not None:
        chunk_t_minus_1 = prev_chunk.reset_index(drop=True).add_suffix('_t-1')
        chunk_t = df_chunk.reset_index(drop=True).add_suffix('_t')

        df_concat = pd.concat([chunk_t_minus_1, chunk_t], axis=1)

        run_and_plot_on_chunk(df_concat)

    prev_chunk = df_chunk