import networkx as nx
import numpy as np
from ltr_wnl.sampling.strategy_base import SamplerStrategy

class GraphBridgeSampler(SamplerStrategy):
    """
    Selects pairs to maximize graph connectivity and global anchoring.
    Uses Betweenness Centrality to identify 'bottleneck' items.

    Parameters
    ----------
    burn_in_count : int
        Number of random comparisons before graph-based sampling
    random_ratio : float, default=0.2
        Not currently used (kept for backwards compatibility)
    """
    def __init__(self, burn_in_count: int, random_ratio: float = 0.2):
        self.burn_in_count = burn_in_count
        self.random_ratio = random_ratio

    def select_next_pair(self, comparison_data, n_items: int) -> tuple[int, int]:
        total_collected = comparison_data.total_comparisons()

        # 1. Burn-in Phase: Random sampling to get baseline connectivity
        if total_collected < self.burn_in_count:
            return self._random_pair(n_items)

        # 2. Build the graph from ComparisonData
        G = nx.DiGraph()
        G.add_nodes_from(range(n_items))
        
        for (i, j), outcomes in comparison_data.comparisons.items():
            if outcomes:
                # Add edge from loser to winner
                if outcomes[-1] == 1: G.add_edge(j, i)
                else: G.add_edge(i, j)

        # 3. Check for disconnected components
        components = list(nx.weakly_connected_components(G))
        if len(components) > 1:
            # Pick one item from two different islands to 'bridge' them
            c1, c2 = np.random.choice(len(components), 2, replace=False)
            u = np.random.choice(list(components[c1]))
            v = np.random.choice(list(components[c2]))
            return tuple(sorted((u, v)))

        # 4. If connected, target nodes with high Betweenness Centrality
        # These are items that sit on the 'path' between others
        centrality = nx.betweenness_centrality(G)

        # Pick a high-centrality node and a random node to refine the 'bridge'
        sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        top_node = sorted_nodes[0][0]
        random_node = np.random.choice(n_items)

        return tuple(sorted((top_node, random_node)))

    def _random_pair(self, n_items: int) -> tuple[int, int]:
        """Select a random pair."""
        i, j = np.random.choice(n_items, size=2, replace=False)
        return tuple(sorted([i, j]))