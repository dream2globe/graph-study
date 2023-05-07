import networkx as nx
import pandas as pd
from networkx.exception import NetworkXError

from src.utils.graph import display_graph


def pagerank(
    G,
    alpha=0.85,
    personalization=None,
    max_iter=100,
    tol=1.0e-6,
    nstart=None,
    weight="value",
    dangling=None,
) -> list:
    """Return the PageRank of the nodes in the graph.
    PageRank computes a ranking of the nodes in the graph G based on
    the structure of the incoming links. It was originally designed as
    an algorithm to rank web pages.
    Parameters
    ----------
    G : graph
        A NetworkX graph.  Undirected graphs will be converted to a directed
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
        Edge data key to use as weight.  If None weights are set to 1.
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
    pagerank : list
        List of nodes sorted by PageRank score
    Notes
    -----
    The eigenvector calculation is done by the power iteration method
    and has no guarantee of convergence.  The iteration will stop
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
        personal_weights = dict.fromkeys(W, 1.0 / N)
    else:
        missing = set(G) - set(personalization)
        if missing:
            raise NetworkXError(
                "Personalization dictionary "
                "must have a value for every node. "
                "Missing nodes %s" % missing
            )
        s = float(sum(personalization.values()))
        personal_weights = dict((k, v / s) for k, v in personalization.items())
    if dangling is None:
        # Use personalization vector if dangling vector not specified
        dangling_weights = personal_weights
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
            x[n] += danglesum * dangling_weights[n] + (1.0 - alpha) * personal_weights[n]
        # check convergence, l1 norm
        err = sum([abs(x[n] - xlast[n]) for n in x])
        if err < N * tol:
            return pd.Series(x).sort_values(ascending=False).index.tolist()
    raise NetworkXError(
        "pagerank: power iteration failed to converge " "in %d iterations." % max_iter
    )


def radiorank(G: nx.graph, alpha: float, weight="value", visualization=False, color="#dd4b39"):
    """Pagerank-inspired algorithm is being used to rank the importance of individual test items
        in the radio frequency testing process used in mobile manufacturing.
    Args:
        G (nx.graph): A directed graph using NetworkX.
        alpha (float): Damping parameter
        weight (str): The weight key of a graph with NetworkX
        visualization (bool, optional): If set to True, Node selection order saved in html file. Defaults to False.
    Returns:
        list: The list of selected nodes, which are ordered by importance.
    """
    selected_nodes = []
    num_nodes = G.number_of_nodes()
    for i in range(num_nodes * 2):  # 충분하게 반복할 수 있도록 설정
        # Check if next loop should run or not
        remains = set(G) - set(selected_nodes)
        if not remains:
            break
        personal_weights = dict.fromkeys(G, 0)
        # Set penality
        if selected_nodes:  # if not empty
            for n in personal_weights:
                for nbr in G[n]:
                    if n in selected_nodes:
                        personal_weights[nbr] += G[nbr][n][weight]
            initial_penalty_value = 0.1**10
            sum_personal_weights = sum(personal_weights.values())
            for k, v in personal_weights.items():
                if k in selected_nodes:
                    personal_weights[k] = 0
                else:
                    personal_weights[k] = initial_penalty_value ** (v / sum_personal_weights)
        else:
            personal_weights = dict.fromkeys(G, 1)
        # Scoring with personalization
        sorted_nodes_by_pr = pagerank(
            G, alpha=alpha, weight=weight, personalization=personal_weights
        )
        # Append the most important feature
        for node in sorted_nodes_by_pr:
            if node not in selected_nodes:
                selected_nodes.append(node)
                break
        # Display graph using pyvis lib.
        if visualization:
            G[nbr][n]["color"] = color  # color is red
            display_graph(G, path=f"data/visualization/html/nx_{i:02d}.html", figsize="1500px")
    return selected_nodes
