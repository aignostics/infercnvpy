from functools import cached_property
from typing import Literal

import networkit as nk
import numpy as np
import pandas as pd
from scipy import sparse
from anndata import AnnData

class LouvainCommunityDetection:
    """Louvain community detection wrapper around networkit library with parallelization.

    This class applies the Louvain algorithm to a graph representation of the data, identifying
    communities based on the resolution parameter and additional optional parameters passed to
    networkit's `community.PLM` function. The Louvain algorithm is a greedy optimization method
    intended to extract non-overlapping communities from large networks.
    More info about Louvain algorithm: https://en.wikipedia.org/wiki/Louvain_method.

    This wrapper uses networkit's implementation, a parallel, high-performance version of Louvain
    with additional options such as refinement, multi-resolution support, and recursive coarsening.
    Can handle much larger graphs (millions to billions of edges) efficiently because the core
    computation is in optimized C++ with parallel threads.
    """

    def __init__(
        self,
        resolution: float = 2.0,
        refine: bool = True,
        par: Literal["none", "simple", "balanced", "none randomized"] = "none",
    ) -> None:
        """Initialize Louvain community detection.

        Args:
        ----
          resolution: Resolution parameter for the Louvain algorithm. Higher values result in more
            and smaller clusters.
          refine: Whether to perform refinement after the initial partitioning. Refinement can
            improve the quality of the detected communities but may increase computation time.
          par: Parallelization strategy. Choose based on the available computational resources and
            the size of the graph. Available options are:
            - 'none': Algorithm runs sequentially. Useful for debugging or single-threaded
              execution. Only option that guarantees deterministic results.
            - 'simple': A basic strategy where threads divide tasks without special load balancing.
              May be faster for uniform workloads.
            - 'balanced': Attempts to distribute work evenly across threads to avoid idle time.
              Often provides the best trade-off between workload distribution and overhead.
            - 'none randomized': Likely runs sequentially while introducing randomization into
              node/move ordering to reduce bias or contention. Useful if reproducibility or random
              ordering is desired without parallel threads.
        """
        self.resolution = resolution
        self.refine = refine
        self.par = par
        self._x: np.ndarray | sparse.spmatrix

    @cached_property
    def _graph(self) -> nk.Graph:
        """Compute and cache the graph between elements.

        The graph represents the connection between elements and is computed from a 2D array. The
        2D array is interpreted as an adjacency matrix defining the graph.

        Returns
        -------
        The computed graph.
        """
        # Convert to COO format for efficient iteration
        connectivities = sparse.coo_matrix(self._x) if isinstance(self._x, np.ndarray) else self._x.tocoo()

        # Build graph and add edges
        graph = nk.Graph(n=connectivities.shape[0], weighted=True, directed=False)

        # Filter to upper triangle and add edges to avoid duplicates
        mask = connectivities.row < connectivities.col
        for i, j, w in zip(
            connectivities.row[mask],
            connectivities.col[mask],
            connectivities.data[mask],
        ):
            graph.addEdge(int(i), int(j), float(w))

        return graph

    def preprocess(self, X: np.ndarray | pd.DataFrame | sparse.spmatrix) -> nk.Graph:  # noqa: N803 - sklearn styled interface
        """Construct a graph network from input data.

        Args:
        ----
          X: Input data to cluster which is interpreted as a 2D adjacency matrix
            (n_samples, n_samples).

        Returns:
        -------
        The computed graph.
        """
        # Convert pandas dataframe to numpy array if necessary
        x_ = X.to_numpy() if isinstance(X, pd.DataFrame) else X

        # Update internal state only if data has changed
        should_update = False
        if not hasattr(self, "_x"):
            should_update = True
        elif x_.shape != self._x.shape:
            should_update = True
        elif sparse.issparse(x_) and sparse.issparse(self._x):
            # Both sparse - compare using nnz
            should_update = (x_ != self._x).nnz != 0
        elif sparse.issparse(x_) or sparse.issparse(self._x):
            # One sparse, one dense - convert and compare
            if sparse.issparse(x_):
                should_update = (x_ != sparse.csr_matrix(self._x)).nnz != 0
            else:
                should_update = (sparse.csr_matrix(x_) != self._x).nnz != 0
        else:
            # Both dense - use array_equal
            should_update = not np.array_equal(x_, self._x)

        if should_update:
            # Clear cached graph property
            if "_graph" in self.__dict__:
                del self._graph
            self._x = x_

        # Return cached or newly computed graph
        return self._graph

    def fit(
        self,
        X: nk.Graph,  # noqa: N803 - sklearn styled interface
        **kwargs,
    ) -> tuple[pd.Categorical, float | None]:
        """Fit Louvain community detection model to data graph.

        Performs Louvain community detection on the input graph and returns cluster labels and
        modularity of clusters.

        Args:
        ----
          X: The graph instance as returned by `preprocess`.
          **kwargs: The keyword arguments `resolution`, `refine`, `par` or any additional keyword
            arguments to pass to networkit's `community.PLM` function. For reference:
            https://networkit.github.io/dev-docs/python_api/community.html#networkit.community.PLM

        Returns:
        -------
        Cluster labels for each sample and modularity. If the graph has no edges, modularity is
        None.
        """
        # Override the default parameters if passed on
        self.resolution = kwargs.pop("resolution", self.resolution)
        self.refine = kwargs.pop("refine", self.refine)
        self.par = kwargs.pop("par", self.par)

        plm = nk.community.PLM(
            X, refine=self.refine, gamma=self.resolution, par=self.par, **kwargs
        )
        plm.run()
        partition = plm.getPartition()

        # Only compute modularity if there are edges in the graph, otherwise return None
        modularity = None
        if X.numberOfEdges() > 0:
            modularity = nk.community.Modularity().getQuality(partition, X)

        return pd.Categorical(partition.getVector()), modularity

    def fit_predict(self, X: np.ndarray | pd.DataFrame | sparse.spmatrix, **kwargs) -> pd.Categorical:  # noqa: N803 - sklearn styled interface
        """Compute cluster centers and predict cluster index for each sample.

        Convenience method that calls `preprocess` and `fit` and returns the cluster labels.

        Args:
        ----
          X: Input data to cluster, array-like of shape (n_samples, n_samples).
          **kwargs: Additional keyword arguments to pass to the fit method.

        Returns:
        -------
        Cluster labels for each sample.
        """
        graph = self.preprocess(X)
        y_clusters, _ = self.fit(graph, **kwargs)
        return y_clusters