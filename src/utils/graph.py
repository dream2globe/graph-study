import time
from itertools import combinations
from pathlib import Path

import imageio
import networkx as nx
import pandas as pd
from PIL import Image
from playwright.sync_api import sync_playwright
from pyvis.network import Network

from src.utils.logger import get_logger

logger = get_logger()
random_seed = 42


def build_nx_graph(
    corr_mat: pd.DataFrame,
    titles: list[str],
    pos: pd.DataFrame | None = None,
    highlights: list[str] | None = None,
    threshold: float = 0.0,
) -> nx.Graph:
    """Build a graph using Python's NetworkX library
    Args:
        corr_mat (pd.DataFrame): A matrix representing the correlation values between paired nodes
        titles (list[str]): Node titles
        pos (pd.DataFrame | None, optional): Location of nodes in a graph. Defaults to None.
        highlights (list[str] | None, optional): Highlighted node with a specific color. Defaults to None.
        threshold (float, optional): _description_. Defaults to 0.0.
    Returns:
        nx.Graph: a directed graph with the NetworkX library
            to model the relationships between nodes in RF test items
    """
    G = nx.Graph()
    for i, title in enumerate(titles):
        G.add_node(i, label=f"Node {i}", title=title)
    for n1, n2 in combinations(G.nodes, 2):
        if abs(corr_mat.loc[n1, n2]) < threshold:  # there is no edges btw node1 and node2
            continue
        G.add_edges_from([(n1, n2, {"value": abs(corr_mat.loc[n1, n2])})])
    if pos is not None:
        assert G.number_of_nodes() == len(pos)
        for row in pos.itertuples():
            G.nodes[row.Index]["x"], G.nodes[row.Index]["y"] = row.x, row.y
    if highlights is not None:
        for node in highlights:
            G.nodes[node]["color"] = "#dd4b39"
    return G.to_directed()


def display_graph(nx_graph, path, figsize="1500px"):
    net = Network(figsize, figsize)
    net.force_atlas_2based()
    net.from_nx(nx_graph)
    net.write_html(path)


def html2ani(source_path: Path, save_file: Path) -> None:
    temp = Path().cwd() / "temp"
    logger.info("Converting HTML to PNG")
    html_files = source_path.glob("*.html")
    for html in html_files:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False, args=["--start-maximized"])
            page = browser.new_page()
            page.goto(
                f"file://{html}",
                timeout=60000,
                wait_until="networkidle",
            )
            time.sleep(0.5)
            temp_file = temp / f"{html.stem}.png"
            page.screenshot(path=temp_file, full_page=True)
            browser.close()
    logger.info("Converting PNG to animated GIF")
    png_files = [str(path) for path in temp.glob("*.png")]
    pngs = [Image.open(png) for png in sorted(png_files)]
    imageio.mimsave(save_file, pngs, "GIF", fps=2)
    logger.info("finished.")
