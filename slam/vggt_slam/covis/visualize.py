from __future__ import annotations
import os, math
from typing import Dict
import matplotlib.pyplot as plt
import networkx as nx

def _fmt_pose(p):
    tx,ty,tz,qx,qy,qz,qw = p
    return f"t=({tx:.2f},{ty:.2f},{tz:.2f})\nq=({qx:.2f},{qy:.2f},{qz:.2f},{qw:.2f})"

def draw_graph_png(G: nx.DiGraph, out_png: str):
    """
    Renders a labeled graph PNG:
      - Node label: submap id, DEM bbox, pose
      - Parent edges: solid green, Similarity edges: dotted orange (thickness ∝ sim)
    """
    if G.number_of_nodes() == 0:
        return

    plt.figure(figsize=(12, 8), dpi=150)
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot")  # requires graphviz
    except Exception:
        pos = nx.spring_layout(G, k=1.2 / math.sqrt(max(1, G.number_of_nodes())), seed=42)

    labels = {}
    for n, data in G.nodes(data=True):
        bbox = data.get("dem_bbox")
        bbox_s = f"{bbox}" if bbox else "None"
        pose = data.get("pose", (0,0,0,0,0,0,1))
        hdr = f"S{n}"
        lbl = f"{hdr}\nDEM:{bbox_s}\n{_fmt_pose(pose)}"
        labels[n] = lbl

    parent_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("kind") == "parent"]
    sim_edges = [(u, v, d.get("sim", 0.0)) for u, v, d in G.edges(data=True) if d.get("kind") == "sim"]

    nx.draw_networkx_nodes(G, pos, node_color="#e2f0ff", edgecolors="#1f77b4", linewidths=1.5, node_size=1400)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=7)

    if parent_edges:
        nx.draw_networkx_edges(G, pos, edgelist=parent_edges, width=2.5, arrows=True,
                               arrowstyle="-|>", edge_color="#2ca02c")

    if sim_edges:
        widths = [max(0.5, 3.0 * s) for (_, _, s) in sim_edges]
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v) for (u, v, _) in sim_edges],
                               width=widths, style="dotted", arrows=False, edge_color="#ff7f0e")

    plt.axis("off")
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
