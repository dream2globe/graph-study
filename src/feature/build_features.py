from itertools import combinations

import networkx as nx
import pandas as pd
from networkx.exception import NetworkXError

from logger import get_logger

logger = get_logger()
random_seed = 42


def build_nx_graph(corr_mat: pd.DataFrame, threshold: float = 0.0):
    """Convert correlation matrix to directed graph
    Args:
        corr_mat (pd.DataFrame): Matrix to contain correlation values between paired nodes
        threshold (float): Minimal value to connect between edges
    Returns:
        nx.Graph: NetworkX graph as correlation matrix
    """
    G = nx.Graph()
    nodes = corr_mat.columns.tolist()
    G.add_nodes_from(nodes)

    for n1, n2 in combinations(G.nodes, 2):
        if abs(corr_mat.loc[n1, n2]) < threshold:  # there is no edges btw node1 and node2
            continue
        G.add_edges_from([(n1, n2, {"value": abs(corr_mat.loc[n1, n2])})])
    return G.to_directed()


def pagerank(
    G,
    alpha=0.85,
    personalization=None,
    max_iter=100,
    tol=1.0e-6,
    nstart=None,
    weight="weight",
    dangling=None,
):
    """Return the PageRank of the nodes in the graph.
    PageRank computes a ranking of the nodes in the graph G based on
    the structure of the incoming links. It was originally designed as
    an algorithm to rank web pages.

    Parameters
    ----------
    G : graph
    A NetworkX graph. Undirected graphs will be converted to a directed
    graph with two directed edges for each undirected edge.

    alpha : float, optional
    Damping parameter for PageRank, default=0.85.

    personalization: dict, optional
    The "personalization vector" consisting of a dictionary with a
    key for every graph node and nonzero personalization value for each node.
    By default, a uniform distribution is used.

    max_iter : integer, optional
    Maximum number of iterations in power method eigenvalue solver.

    tol : float, optional
    Error tolerance used to check convergence in power method solver.

    nstart : dictionary, optional
    Starting value of PageRank iteration for each node.

    weight : key, optional
    Edge data key to use as weight. If None weights are set to 1.

    dangling: dict, optional
    The outedges to be assigned to any "dangling" nodes, i.e., nodes without
    any outedges. The dict key is the node the outedge points to and the dict
    value is the weight of that outedge. By default, dangling nodes are given
    outedges according to the personalization vector (uniform if not
    specified). This must be selected to result in an irreducible transition
    matrix (see notes under google_matrix). It may be common to have the
    dangling dict to be the same as the personalization dict.

    Returns
    -------
    pagerank : dictionary
    Dictionary of nodes with PageRank as value

    Notes
    -----
    The eigenvector calculation is done by the power iteration method
    and has no guarantee of convergence. The iteration will stop
    after max_iter iterations or an error tolerance of
    number_of_nodes(G)*tol has been reached.

    The PageRank algorithm was designed for directed graphs but this
    algorithm does not check if the input graph is directed and will
    execute on undirected graphs by converting each edge in the
    directed graph to two edges.


    """
    if len(G) == 0:
        return {}
    if not G.is_directed():
        D = G.to_directed()
    else:
        D = G
    # Create a copy in (right) stochastic form
    W = nx.stochastic_graph(D, weight=weight)
    N = W.number_of_nodes()
    # Choose fixed starting vector if not given
    if nstart is None:
        x = dict.fromkeys(W, 1.0 / N)
    else:
        # Normalized nstart vector
        s = float(sum(nstart.values()))
        x = dict((k, v / s) for k, v in nstart.items())
    if personalization is None:
        # Assign uniform personalization vector if not given
        p = dict.fromkeys(W, 1.0 / N)
    else:
        missing = set(G) - set(personalization)
        if missing:
            raise NetworkXError(
                "Personalization dictionary "
                "must have a value for every node. "
                "Missing nodes %s" % missing
            )
        s = float(sum(personalization.values()))
        p = dict((k, v / s) for k, v in personalization.items())
    if dangling is None:
        # Use personalization vector if dangling vector not specified
        dangling_weights = p
    else:
        missing = set(G) - set(dangling)
        if missing:
            raise NetworkXError(
                "Dangling node dictionary "
                "must have a value for every node. "
                "Missing nodes %s" % missing
            )
        s = float(sum(dangling.values()))
        dangling_weights = dict((k, v / s) for k, v in dangling.items())
    dangling_nodes = [n for n in W if W.out_degree(n, weight=weight) == 0.0]
    # power iteration: make up to max_iter iterations
    for _ in range(max_iter):
        xlast = x
        x = dict.fromkeys(xlast.keys(), 0)
        danglesum = alpha * sum(xlast[n] for n in dangling_nodes)
        for n in x:
            # this matrix multiply looks odd because it is
            # doing a left multiply x^T=xlast^T*W
            for nbr in W[n]:
                x[nbr] += alpha * xlast[n] * W[n][nbr][weight]
            x[n] += danglesum * dangling_weights[n] + (1.0 - alpha) * p[n]
        # check convergence, l1 norm
        err = sum([abs(x[n] - xlast[n]) for n in x])
        if err < N * tol:
            return x
    raise NetworkXError(
        "pagerank: power iteration failed to converge " "in %d iterations." % max_iter
    )


def testrank(
    G: nx.Graph,
    alpha=0.85,
    personalization=None,
    max_iter=100,
    tol=1.0e-6,
    nstart=None,
    weight="weight",
    dangling=None,
    init_penalty=0.5,
    velocity=None,
):
    """_summary_

    Args:
        G (nx.Graph): A NetworkX graph.  Undirected graphs will be converted to a directed
            graph with two directed edges for each undirected edge.
        alpha (float, optional): Damping parameter for PageRank. Defaults to 0.85.
        personalization (dict, optional): The "personalization vector" consisting of a dictionary with a
            key for every graph node and nonzero personalization value for each node.
            By default, a uniform distribution is used. Defaults to None.
        max_iter (int, optional): Maximum number of iterations in power method eigenvalue solver. Defaults to 100.
        tol (float, optional): Error tolerance used to check convergence in power method solver. Defaults to 1.0e-6.
        nstart (dict, optional): Starting value of PageRank iteration for each node. Defaults to None.
        weight (str, optional): Edge data key to use as weight. Defaults to "weight".
        dangling (dict, optional): The outedges to be assigned to any "dangling" nodes, i.e., nodes without
            any outedges. The dict key is the node the outedge points to and the dict value is the weight
            of that outedge. By default, dangling nodes are given outedges according to the personalization
            vector (uniform if not specified). This must be selected to result in an irreducible transition
            matrix (see notes under google_matrix). It may be common to have the dangling dict to be the
            same as the personalization dict. Defaults to None.
        init_penalty (int, optional): Initial penalty score. Defaults to 0.5.
        velocity (dict, optional): Speed of convergence to 1 of penalty score. Defaults to None.

    Returns:
        dict: Dictionary of nodes with PageRank as value
    """
    # Convert to a directed graph
    if not G.is_directed():
        D = G.to_directed()
    else:
        D = G
    # Create a copy in (right) stochastic form
    W = nx.stochastic_graph(D, weight="initial")
    N = W.number_of_nodes()
    # Set initial PR scores
    if nstart is None:
        x = dict.fromkeys(W, 1.0 / N)  # Choose fixed starting vector if not given
    else:
        s = float(sum(nstart.values()))  # Normalized nstart vector
        x = dict((k, v / s) for k, v in nstart.items())
    # Set personalization
    if personalization is None:
        p = dict.fromkeys(W, 1.0 / N)  # Assign uniform personalization vector if not given
    else:
        missing = set(G) - set(personalization)
        if missing:
            raise nx.NetworkXError(
                f"Personalization dictionary must have a value for every node. nodes {missing}"
            )
        s = float(sum(personalization.values()))
        p = dict((k, v / s) for k, v in personalization.items())
    # Handling with dangling
    if dangling is None:
        dangling_weights = p  # Use personalization vector if dangling vector not specified
    else:
        missing = set(G) - set(dangling)
        if missing:
            raise nx.NetworkXError(
                f"Dangling node dictionary must have a value for every node. nodes {missing}"
            )
        s = float(sum(dangling.values()))
        dangling_weights = dict((k, v / s) for k, v in dangling.items())
    dangling_nodes = [n for n in W if W.out_degree(n, weight=weight) == 0.0]
    selected_nodes = []
    for _ in range(len(G)):
        # Initailize penality terms
        penalty_rates = dict.fromkeys(G, 1)
        for sel_n in selected_nodes:
            for nbr in G[sel_n]:
                penalty_rates[nbr] *= init_penalty ** G[sel_n][nbr][weight]
        # power iteration: make up to max_iter iterations
        for i in range(max_iter):
            xlast = x
            x = dict.fromkeys(xlast.keys(), 0)
            danglesum = alpha * sum(xlast[n] for n in dangling_nodes)
            for n in x:
                penalty = (
                    penalty_rates[n] if velocity is None else penalty_rates[n] ** (i / velocity)
                )
                for nbr in W[n]:
                    x[nbr] += alpha * xlast[n] * W[n][nbr]["initial"] * penalty
                x[n] += danglesum * dangling_weights[n] + (1.0 - alpha * penalty) * p[n]
            # check convergence, l1 norm
            err = sum([abs(x[n] - xlast[n]) for n in x])
            if err < N * tol:
                break
        # select the best score & new node
        ordered_nodes = pd.Series(x).sort_values(ascending=True).keys()
        for ord_n in ordered_nodes:
            if ord_n not in selected_nodes:
                selected_nodes.append(ord_n)
                break
    return selected_nodes
