import numpy as np
import pandas as pd
import pip

def calc_preds(self, aData):
    preds = []
    sc.pp.neighbors(aData)
    print("neighbors are calculating!")
    for i in range (aData.shape[0]):
        nbrs = aData.obsp["connectivities"][i].tocoo().col
        pred = max(set(exp_data.whole[nbrs].obs["cluster_s"]), key = exp_data.whole[nbrs].obs["cluster_s"].tolist().count)
        preds.append(pred)
    return np.array(preds)


def Harmony(exp_data):
    try:
        import harmonypy as hm
    except ModuleNotFoundError:
        pip.main(['install', 'harmonypy'])
        
    sc.pp.pca(exp_data.whole)
    new_vals = hm.run_harmony(exp_data.whole.obsm["X_pca"], exp_data.whole.obs, "batch_id", max_iter_harmony = 20)
    res = pd.DataFrame(new_vals.Z_corr)
    res = res.T
    exp_data.whole.obsm["Harmony"] = res.to_numpy()
    if "cluster_s" in exp_data.whole.obs:
        _adata = sc.AnnData(X= res, obs = exp_data.whole.obs, var = pd.DataFrame(res.columns)) 
        exp_data.whole.obs["Harmony-preds"] = calc_preds(_adata)

def MNN(exp_data):  
    try:
        import mnnpy
    except ModuleNotFoundError:
        pip.main(['install', 'mnnpy'])
    
    hvg = exp_data.whole.var.index.tolist()
    _datasets = []
    for db in exp_data.dataset_list:
        _db = exp_data.dataset_list
        _db.obs.index = _db.obs.index.map(int)
        _datasets.append(_db)
    corrected  = mnnpy.mnn_correct(_datasets, var_subset = hvg)
    whole_data = []
    for res in corrected[0]:
        whole_data.extend(res.X)
    exp_data.whole.obsm["MNN"] = np.array(whole_data)
    if "cluster_s" in exp_data.whole.obs:
        _adata = exp_data.whole.copy() 
        _adata.X = np.array(whole_data)
        exp_data.whole.obs["MNN-preds"] = calc_preds(_adata)

def Scanorama(exp_data):
    try:
        import scanorama as sca
    except ModuleNotFoundError:
        pip.main(['install', 'scanorama'])
        
    corrected = sca.correct_scanpy(exp_data.dataset_list, return_dimred=True)
    whole_data = []
    for ds in corrected:
        ds.X = ds.X.toarray()
        whole_data.extend(ds.X)
    exp_data.whole.obsm["Scanorama"] = np.array(whole_data)
    if "cluster_s" in exp_data.whole.obs:
        _adata = exp_data.whole.copy() 
        _adata.X = np.array(whole_data)
        exp_data.whole.obs["Scanorama-preds"] = calc_preds(_adata)

def SCVI(exp_data):
    try:
        import scvi
    except ModuleNotFoundError:
        pip.main(['install', 'scvi-tools'])
    
    sc_copy = exp_data.whole.copy()
    sc.pp.filter_genes(sc_copy, min_counts=3)
    sc_copy.layers["counts"] = sc_copy.X.copy() 
    sc.pp.normalize_total(sc_copy, target_sum=1e4)
    sc.pp.log1p(sc_copy)
    sc_copy.raw = sc_copy
    sc.pp.highly_variable_genes(
    sc_copy,
    n_top_genes=sc_copy.shape[1],
    subset=True,
    layer="counts",
    flavor="seurat_v3",
    batch_key="batch_id"
    )
    scvi.model.SCVI.setup_anndata(sc_copy, layer="counts", batch_key = "batch_id")
    model = scvi.model.SCVI(sc_copy)
    model.train()
    latent = model.get_latent_representation()
    exp_data.whole.obsm["SCVI"] = latent
    if "cluster_s" in exp_data.whole.obs:
        _adata = sc.AnnData(X= latent, obs = exp_data.whole.obs, var = pd.DataFrame(pd.DataFrame(latent).columns)) 
        exp_data.whole.obs["SCVI-preds"] = calc_preds(_adata)

def MARIO(exp_data):
    try:
        from mario.match import pipelined_mario
    except ModuleNotFoundError:
        pip.main(['install', 'pyMARIO'])
        
    n_datasets = len(exp_data.dataset_list)
    _dataset_list = []
    ds_size = 0
    for ds in exp_data.dataset_list:
        tmp_ds = ds.to_df()
        _dataset_list.append(tmp_ds)
        ds_size = ds.shape[0]
    fm, embedding_lst = pipelined_mario(data_lst = _dataset_list,
                                        normalization=True, n_batches=4,
                                        n_matched_per_cell=1, sparsity_ovlp=500, sparsity_all=500,
                                        n_components_ovlp=15, n_components_all=15,
                                        n_cancor=5, n_wts=4,
                                        n_clusters=10, n_components_filter=10, bad_prop=0.2, max_iter_filter=20,
                                        knn=False, embed_dim=10, max_iter_embed=500, verbose=False)

    emb_size = len(embedding_lst[0][0])
    for lst in embedding_lst:
        print(len(lst))
    print(fm[0][0])
    print(type(embedding_lst[0][0]), embedding_lst[0][0])
    succes_matches = []
    for i in range(len(fm[0])):  
        control_flag = 1
        for j in range(n_datasets):
            if len(fm[j][i]) == 0:
                control_flag = 0
                break
        if control_flag:
            tmp_matches = []
            for j in range(n_datasets):
                tmp_matches.append(fm[j][i][0])
            succes_matches.append(tmp_matches)

    emb_datasets = np.zeros((n_datasets, ds_size, emb_size))
    for i in range(len(succes_matches)):
        for j in range(n_datasets):
            emb_datasets[j, succes_matches[i][j]] = embedding_lst[j][i]
    emb_final = np.vstack(emb_datasets)
    exp_data.whole.obsm["MARIO"] = emb_final
    if "cluster_s" in exp_data.whole.obs:
        _adata = sc.AnnData(X= emb_final, obs = exp_data.whole.obs, var = pd.DataFrame(pd.DataFrame(emb_final).columns)) 
        exp_data.whole.obs["MARIO-preds"] = calc_preds(_adata)
        
def integrate(method, _adata):
    if method == "Harmony":
        Harmony(_adata)
    elif method == "Scanorama":
        Scanorama(_adata)
    elif method == "MARIO":
        MARIO(_adata)
    elif method == "MNN":
        MNN(_adata)
    elif method == "SCVI":
        SCVI(_adata)
    elif method == "All":
        Harmony(_adata)
        Scanorama(_adata)
        MARIO(_adata)
        MNN(_adata)
        SCVI(_adata)
    else:
        print("The given integration method("+ method +") is not available.")