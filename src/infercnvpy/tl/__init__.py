from typing import Union

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scanpy import logging

from ._clustering import LouvainCommunityDetection
from ._copykat import copykat
from ._infercnv import infercnv
from ._scores import cnv_score, ithcna, ithgex


def louvain(
    adata: AnnData,
    resolution: float = 2.0,
    key_added: str = "cnv_cluster",
    neighbors_key: str = "cnv_neighbors",
    **kwargs,
) -> pd.Categorical | None:
    """Perform Louvain community detection on CNV neighborhood graph.
    
    Parameters
    ----------
    adata
        AnnData object with computed CNV neighbors
    resolution
        Resolution parameter for Louvain algorithm. Higher values = more clusters
    key_added
        Key to store cluster labels in adata.obs
    neighbors_key
        Key for the neighbors connectivity matrix in adata.obsp
    **kwargs
        Additional arguments passed to LouvainCommunityDetection
        
    Returns
    -------
    If inplace, returns None. Otherwise returns cluster labels.
    """
    connectivity_key = f"{neighbors_key}_connectivities"
    
    if connectivity_key not in adata.obsp:
        raise ValueError(
            f"{connectivity_key} not found in adata.obsp. "
            f"Run infercnvpy.pp.neighbors() first."
        )
    
    louvain = LouvainCommunityDetection(resolution=resolution, **kwargs)
    clusters = louvain.fit_predict(adata.obsp[connectivity_key])
    
    adata.obs[key_added] = clusters
    return None


def pca(
    adata: AnnData,
    svd_solver: str = "arpack",
    zero_center: bool = False,
    inplace: bool = True,
    use_rep: str = "cnv",
    key_added: str = "cnv_pca",
    **kwargs,
) -> np.ndarray | None:
    """Compute the PCA on the result of :func:`infercnvpy.tl.infercnv`.

    Thin wrapper around :func:`scanpy.pp.pca`.

    Parameters
    ----------
    adata
        annotated data matrix
    svd_solver
        See :func:`scanpy.pp.pca`.
    zero_center
        See :func:`scanpy.pp.pca`.
    inplace
        If True, store the result in adata.obsm. Otherwise return the PCA matrix.
    use_rep
        Key under which the result of infercnv is stored in adata
    key_added
        Key under which the result will be stored in adata.obsm if `inplace=True`.
    **kwargs
        Additional arguments passed to :func:`scanpy.pp.pca`.
    """
    if f"X_{use_rep}" not in adata.obsm:
        raise KeyError(f"X_{use_rep} is not in adata.obsm. Did you run `tl.infercnv`?")

    pca_res = sc.tl.pca(
        adata.obsm[f"X_{use_rep}"],
        svd_solver=svd_solver,
        zero_center=zero_center,
        **kwargs,
    )
    if inplace:
        adata.obsm[f"X_{key_added}"] = pca_res
    else:
        return pca_res


def umap(
    adata: AnnData,
    neighbors_key: str = "cnv_neighbors",
    key_added: str = "cnv_umap",
    inplace: bool = True,
    **kwargs,
):
    """Compute the UMAP on the result of :func:`infercnvpy.tl.infercnv`.

    Thin wrapper around :func:`scanpy.tl.umap`

    Parameters
    ----------
    adata
        annotated data matrix
    neighbors_key
        Key under which the result of :func:`infercnvpy.pp.neighbors` is stored
        in adata
    key_added
        Key under which the result UMAP will be stored in adata.obsm
    inplace
        If True, store the result in adata.obsm, otherwise return the result of UMAP.
    **kwargs
        Additional arguments passed to :func:`scanpy.tl.umap`.
    """
    tmp_adata = sc.tl.umap(adata, neighbors_key=neighbors_key, copy=True, **kwargs)

    if inplace:
        adata.obsm[f"X_{key_added}"] = tmp_adata.obsm["X_umap"]
    else:
        return tmp_adata.obsm["X_umap"]


def tsne(
    adata: AnnData,
    use_rep: str = "cnv_pca",
    key_added: str = "cnv_tsne",
    inplace: bool = True,
    **kwargs,
):
    """Compute the t-SNE on the result of :func:`infercnvpy.tl.infercnv`.

    Thin wrapper around :func:`scanpy.tl.tsne`

    Parameters
    ----------
    adata
        annotated data matrix
    use_rep
        Key under which the result of :func:`infercnvpy.tl.pca` is stored
        in adata
    key_added
        Key under which the result of t-SNE will be stored in adata.obsm
    inplace
        If True, store the result in adata.obsm, otherwise return the result of t-SNE.
    **kwargs
        Additional arguments passed to :func:`scanpy.tl.tsne`.
    """
    if f"X_{use_rep}" not in adata.obsm and use_rep == "cnv_pca":
        logging.warning("X_cnv_pca not found in adata.obsm. Computing PCA with default parameters")  # type: ignore
        pca(adata)
    tmp_adata = sc.tl.tsne(adata, use_rep="X_cnv_pca", copy=True, **kwargs)

    if inplace:
        adata.obsm[f"X_{key_added}"] = tmp_adata.obsm["X_tsne"]
    else:
        return tmp_adata.obsm["X_tsne"]
