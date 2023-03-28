import numpy as np
import os
import scanpy as sc, anndata as ad
import pandas as pd
from sklearn.preprocessing import Normalizer

import sys
sys.path.insert(1, 'src')
from comparison import integrate
from scpro_hi import horizontal_integration
from performance import EVAL, umap_projection
class scData():
    dataset_list = []
    whole = None
    def __init__(self, base_path, concat = True, _load = False, sub_sample = False):
        directory = os.fsencode(base_path)
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if filename.endswith(".h5ad") or filename.endswith(".h5"):
                print(filename + " is under process.")
                if filename.endswith(".h5ad"):
                    tmp_ds = sc.read_h5ad(base_path + "/" + filename)
                else:
                    tmp_ds = sc.read_hdf(base_path + "/" + filename, "matrix")
                tmp_ds.obs["batch_id"] = np.repeat(os.path.basename(filename).replace(".h5ad","").replace(".h5",""), tmp_ds.shape[0])
                tmp_ds.uns["name"] = os.path.basename(filename).replace(".h5ad","").replace(".h5","")
                if isinstance(tmp_ds.X, np.ndarray) == False:
                    tmp_ds.X = tmp_ds.X.toarray()
                if sub_sample:
                    if _load:
                        rnd_cell_list = np.load(base_path + "/Subsamples/" + tmp_ds.uns["name"]+".npy")
                    else:
                        rnd_cell_list = np.random.choice(tmp_ds.shape[0], 20000, replace=False)
                        np.save(base_path + "/Subsamples/" + tmp_ds.uns["name"]+".npy", rnd_cell_list)
                    tmp_ds = tmp_ds[rnd_cell_list, 1:]

                if len(tmp_ds.obs.index) > 0:
                    self.dataset_list.append(tmp_ds)
                else:
                    print(filename + " is empty.")
        
        self.pre_process()
        if concat:
            self.whole = ad.concat(self.dataset_list, join="inner")
            self.whole.obs_names_make_unique()
            self.whole.uns["name"] = "Whole"
            for i in range(len(self.dataset_list)):
                self.dataset_list[i] = self.dataset_list[i][:,self.whole.var.index]
    def save(self):
        for ds in self.dataset_list:
            ds.write("sub_data/" + ds.uns["name"] + ".h5ad")
            
    def pre_process(self):
        for ds in self.dataset_list:
            ds.X = np.nan_to_num(ds.X)
            self.L2_norm(ds)
    def L2_norm(self, ds):
        model = Normalizer(norm='l2').fit(ds.X)
        ds.X = model.transform(ds.X)

def eval_results(_scData, method):
    if _scData.whole.obsm[method].shape[1] == _scData.whole.X.shape[1]:
        new_aData = _scData.whole.copy()
        a_data_copy = _scData.whole.copy()
        new_aData.X = new_aData.obsm[method]
        ex_lab = EVAL(method, new_aData, "cluster_s", method + "-preds", method, annData_2 = a_data_copy,  exp = "All", save = False, verbose = False)
    else:
        _adata = sc.AnnData(X = _scData.whole.obsm[method], obs = _scData.whole.obs, var = pd.DataFrame(pd.DataFrame(_scData.whole.obsm[method]).columns)) 
        _adata.uns["name"] = _scData.whole.uns["name"]
        ex_lab = EVAL(method, _adata, "cluster_s", method + "-preds", method,  exp = "All", save = False, verbose = False)
    return ex_lab.get_results()

def scprohi_run(method, data_path):
    if  '_adata' not in globals():
        _adata = scData(data_path)
    if method == "SCPRO-HI":
        horizontal_integration(_adata)
    elif method == "All":
        horizontal_integration(_adata)
        integrate(method, _adata)
    else:
        integrate(method, _adata)
    return _adata
        
        
def main(method, _adata):
    if method == "SCPRO-HI":
        horizontal_integration(_adata)
    elif method == "All":
        horizontal_integration(_adata)
        integrate(method, _adata)
    else:
        integrate(method, _adata)
    
    


if __name__ == "__main__":
    args = sys.argv
    if  '_adata' not in globals():
        _adata = scData(args[2])
    main(args[1], _adata)
    
