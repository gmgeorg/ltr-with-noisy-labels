"""
PageRank-based ranking algorithm using NetworkX.

Uses the comparison graph structure where edges represent wins,
and applies Google's PageRank algorithm to infer item quality.
"""

import numpy as np
import networkx as nx

from ltr_wnl.comparisons.data import ComparisonData
from ltr_wnl.ranking.ranker_base import Ranker


class PageRankRanker(Ranker):
    """
    Rank items using PageRank on the comparison graph.

    Constructs a directed graph where an edge from i to j indicates
    that item i beat item j in a comparison. Edge weights reflect
    the number of wins. PageRank scores are then used to rank items.

    Parameters
    ----------
    alpha : float, default=0.85
        Damping parameter for PageRank (1 - teleport probability)

    Examples
    --------
    >>> from ltr_wnl.comparisons import ComparisonData
    >>>
    >>> data = ComparisonData(n_items=3)
    >>> data.add_comparison(0, 1, outcome=1)
    >>> data.add_comparison(0, 2, outcome=1)
    >>> data.add_comparison(1, 2, outcome=1)
    >>>
    >>> ranker = PageRankRanker()
    >>> ranking = ranker.fit(data)
    >>> ranking[0]  # Best item should be 0
    0
    """

    def __init__(self, alpha: float = 0.85):
        if not (0 <= alpha <= 1):
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.alpha = alpha

    def fit(self, comparison_data: ComparisonData) -> list[int]:
        """
        Infer ranking using PageRank.

        Parameters
        ----------
        comparison_data : ComparisonData
            Pairwise comparison outcomes

        Returns
        -------
        ranking : list[int]
            Estimated ranking (item indices from best to worst)
        """
        n_items = comparison_data.n_items

        # Build directed graph
        G = nx.DiGraph()
        G.add_nodes_from(range(n_items))

        # Add edges weighted by number of wins
        for (i, j), outcomes in comparison_data.comparisons.items():
            wins_i = sum(1 for o in outcomes if o == 1)
            wins_j = sum(1 for o in outcomes if o == -1)

            # Add edge from winner to loser
            if wins_i > 0:
                if G.has_edge(i, j):
                    G[i][j]['weight'] += wins_i
                else:
                    G.add_edge(i, j, weight=wins_i)

            if wins_j > 0:
                if G.has_edge(j, i):
                    G[j][i]['weight'] += wins_j
                else:
                    G.add_edge(j, i, weight=wins_j)

        # Handle isolated nodes (items with no comparisons)
        # Give them a small default PageRank by adding self-loops
        for node in G.nodes():
            if G.degree(node) == 0:
                G.add_edge(node, node, weight=1.0)

        # Compute PageRank
        pagerank_scores = nx.pagerank(
            G,
            alpha=self.alpha,
            weight='weight',
            max_iter=100,
            tol=1e-6
        )

        # Convert scores to ranking (higher score = better rank)
        scores = np.array([pagerank_scores[i] for i in range(n_items)])
        ranking = np.argsort(scores)[::-1].tolist()

        return ranking

    def __repr__(self) -> str:
        return f"PageRankRanker(alpha={self.alpha})"
