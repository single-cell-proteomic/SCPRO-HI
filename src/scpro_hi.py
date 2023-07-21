from itertools import count
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scanpy as sc
from scipy.spatial import cKDTree
# import shap
from sklearn.cluster import DBSCAN, MeanShift, KMeans, OPTICS
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, f1_score
# import umap.umap_ as umap
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import umap
from VAE import VAE

import mnnpy
import anndata as ad
from sklearn.cross_decomposition import CCA
import bbknn

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def umap_projection(X, Y, name):
    if isinstance(X, pd.DataFrame) == False:
        df = pd.DataFrame(X)
    else:
        df = X.copy()
    category_o = {}
    for lbl in np.unique(Y):
        category_o[lbl] = np.count_nonzero(Y == lbl)
    if df.shape[1] != 2:
        umap_2d = umap.UMAP(n_components=2, init='random', random_state=0)
        proj_2d = umap_2d.fit_transform(df)
    else:
        proj_2d = df
    df[name] = list(Y)
    # print(type(proj_2d))
    fig_2d = px.scatter(
      proj_2d, x=0, y=1, opacity=0.5,
      color=df[name], labels = {'color': name} 
    )
    fig_2d.show()
    
def calc_preds(aData):
    preds = []
    sc.pp.neighbors(aData, n_neighbors = 3)
    print("neighbors are calculating!")
    for i in range (aData.shape[0]):
        nbrs = aData.obsp["connectivities"][i].tocoo().col
        pred = max(set(aData[nbrs].obs["cluster_s"]), key = aData[nbrs].obs["cluster_s"].tolist().count)
        preds.append(pred)
    return np.array(preds)
        
def find_mutual_nn(data1, data2, k1, k2):
    if k1 > min(data1.shape[0], data2.shape[0]) or k2 > min(data1.shape[0], data2.shape[0]):
        k1 = k2 = min(data1.shape[0], data2.shape[0])
    # print(data1.shape[0], data2.shape[0])
    k_index_1 = cKDTree(data1).query(x=data2, k=k1)[1]
    k_index_2 = cKDTree(data2).query(x=data1, k=k2)[1]
    mutual_1 = []
    mutual_2 = []
    for index_2 in range(k2):
        for index_1 in k_index_1[index_2]:
            if index_2 in k_index_2[index_1]:
                mutual_1.append(index_1)
                mutual_2.append(index_2)
    return mutual_1, mutual_2

def find_mutual_nn2(data1, data2, k1, k2):
    if k1 > min(data1.shape[0], data2.shape[0]) or k2 > min(data1.shape[0], data2.shape[0]):
        k1 = k2 = min(data1.shape[0], data2.shape[0])
    mutual_1 = []
    mutual_2 = []
    if k1 != 0:
        k_index_1 = cKDTree(data1).query(x=data2, k=k1, workers = 8)[1]
        k_index_2 = cKDTree(data2).query(x=data1, k=k2, workers = 8)[1]
        if k1 == 1:
            k_index_2 = [k_index_2]
            k_index_1 = [k_index_1]

        edges = []
        edges_r = []
        for index_1 in range(k1):
            curr = [index_1] * k2 
            edges.extend(list(zip(curr, k_index_2[index_1])))
        for index_2 in range(k2):
            curr = [index_2] * k1 
            edges_r.extend(list(zip(k_index_1[index_2], curr)))
        mutuals = list(set(edges).intersection(set(edges_r)))
        if len(mutuals) > 0:
            mutual_1, mutual_2 = zip(*mutuals)
            mutual_1 = list(mutual_1)
            mutual_2 = list(mutual_2)
    return mutual_1, mutual_2

def find_mutual_nn3(query, target):
    mutual_1 = []
    mutual_2 = []
    if query.shape[0] != 0 and target.shape[0] != 0:
        mutual_2 = cKDTree(target).query(x=query, k=1, workers = 8)[1]
        mutual_1 = np.arange(query.shape[0])
    return mutual_1, mutual_2

def calculate_mismatch_statistics(true_labels, predicted_labels):
    
    mismatch_stats = {}

    for i in range(true_labels.shape[0]):
        match_id = true_labels[i] + "-" + predicted_labels[i]
        if match_id not in mismatch_stats:
            mismatch_stats[match_id] = 0
        else:
            mismatch_stats[match_id] += 1 
    sorted_keys = sorted(mismatch_stats, key=lambda x: mismatch_stats[x])
    for match in sorted_keys:
        print(match, " -> ",mismatch_stats[match])
    df = pd.DataFrame(mismatch_stats, index=[0])
    df.to_excel('cell type matchings.xlsx')
        
def get_HVGs_rf(_adata, y, NIP):
    clf = DTC(random_state=42, max_depth = 3)
    x = _adata.X
    clf.fit(x, y)

    
    hvg_treshold = 1
    feature_names = _adata.var.index.values
    important_features_dict = {}
    for idx, val in enumerate(clf.feature_importances_):
        important_features_dict[feature_names[idx]] = val

    sorted_features_list = sorted(important_features_dict,
                            key = important_features_dict.get,
                            reverse = True)
    current_importance = 0
    current_hvg_count = 0
    filtered_hvgs = []
    for i in range(len(sorted_features_list)):
        if current_hvg_count < NIP:
            filtered_hvgs.append(sorted_features_list[i])
            current_hvg_count += 1
            current_importance += important_features_dict[sorted_features_list[i]]
        else:
            break
    filtered_importance_dict = {key: important_features_dict[key] for key in filtered_hvgs}
    return filtered_importance_dict

def get_HVGs(_adata, y, NIP):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.inspection import permutation_importance
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(_adata.shape[1],)))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    y = np.array(y)
    # Train the model
    model.fit(_adata.X, y, epochs=10, batch_size=32, verbose = False)

    def binary_accuracy(estimator, x, y):
        y_pred = estimator.predict(x, verbose = False)
        y_pred_binary = np.where(y_pred >= 0.5, 1, 0)
        return accuracy_score(y, y_pred_binary)

    # Calculate feature importances using permutation feature importance
    result = permutation_importance(
        model,
        _adata.X,
        y,
        n_repeats=1,
        scoring=binary_accuracy,  # Use custom scoring function
        random_state=42
    )

    # Get feature importances and feature names
    importances = result.importances_mean
    importances[importances<0] = 0
    sum_importances = np.sum(importances)
    if sum_importances > 0:
        importances = importances / sum_importances
    
    

    feature_names = _adata.var.index.values
    important_features_dict = {}
    for i in range(feature_names.shape[0]):
        important_features_dict[feature_names[i]] = importances[i]
        
    sorted_features_list = sorted(important_features_dict,
                            key = important_features_dict.get,
                            reverse = True)
    current_importance = 0
    current_hvg_count = 0
    filtered_hvgs = []
    for i in range(len(sorted_features_list)):
        if current_hvg_count < NIP:
            filtered_hvgs.append(sorted_features_list[i])
            current_hvg_count += 1
            current_importance += important_features_dict[sorted_features_list[i]]
        else:
            break
    filtered_importance_dict = {key: important_features_dict[key] for key in filtered_hvgs}
    return filtered_importance_dict
def cluster_distance(c1, c2):
    similarity = 0
    vec1 = []
    vec2 = []
    for prot in c1:
        if prot in c2:
            vec1.append(c1[prot])
            vec2.append(c2[prot])
    if len(vec1) > 0:
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        similarity = dot_product / (norm1 * norm2)
    return similarity


def hvg_sigmas(dt):
    means = np.array(dt.mean(axis=0))
    var = np.array(dt.var(axis=0))
    model = loess(means, var, degree=2)
    model.fit()
    new_var = model.outputs.fitted_values
    new_var = new_var + abs(np.min(new_var))
    sigma = np.sqrt(new_var)
    return new_var, sigma, means

def multiple_regression(_data, important_genes, target_gene_set):
    return MLPRegressor(random_state = 42).fit(_data[:, important_genes].X, _data[:, target_gene_set].X)

def cell_matching_acc(df, mappings):
    ground_labels = []
    matched_labels = []
    for group in mappings:
        ground_labels.extend([i[0] for i in mappings[group]])
        matched_labels.extend([i[1] for i in mappings[group]])
    print(accuracy_score(df[ground_labels].obs["cluster_s"], df[matched_labels].obs["cluster_s"]))
    print("# of matched cells: ", len(ground_labels))
def horizontal_integration(exp_data, NIP = 38, cluster_treshold = 0.75, extend = False):
    
    # if NIP > exp_data.whole.shape[1]:
    NIP = exp_data.whole.shape[1]
    whole_cluster_list = []
    
    cluster_HVGs = {}
    for db in exp_data.dataset_list:
        if "umap" not in db.obsm:
            db.obsm["umap"] = umap.UMAP(n_components = 5, n_jobs = 8).fit_transform(db.X)
            # db.write("Data/sub_" + db.uns["name"] + ".h5ad")
            # sc.pp.neighbors(db)
            # sc.tl.umap(db)
            # db.obsm["umap"] = db.obsm["X_umap"]
            # umap_projection(db.obsm["umap"], db.obs["cluster_s"], db.uns["name"])
#             # db.obsm["umap"] = db.X
#             # db.obsm["umap_features"] = TSNE(n_components = 2).fit_transform(db.X)
#         neigh = NearestNeighbors(n_neighbors=2, metric= "euclidean")
#         nbrs = neigh.fit(db.obsm["umap"])
#         distances, indices = nbrs.kneighbors(db.obsm["umap"])
#         distances = np.sort(distances[:,1], axis=0)
#         plt.plot(distances)
#         x = np.arange(len(distances))
#         _eps = distances[x.argmax()]
#         print(_eps)
#         distances = distances[:,1]
#         dist_diffs = (distances[1:] - distances[:-1])
#         for i in range (1, len(dist_diffs)):
#             if dist_diffs[i] > dist_diffs[i-1] * 2:
#                 curvest = dist_diffs[i]
#                 break
#         _eps = curvest

#         fig = px.scatter(
#             distances, 
#             title='Distance Curve')
#         fig.update_xaxes(title_text='Distances')
#         fig.update_yaxes(title_text='Distance threashold (espsilon)')
#         fig.show()
#             print(_eps)
        
        # model = DBSCAN(eps = 0.35, min_samples = 2).fit(db.obsm["umap"])
        model = MeanShift(bandwidth = 2.0).fit(db.obsm["umap"])
        cluster_ids = np.array([str(x) + "_" + db.uns["name"] for x in model.labels_])
        clusters, cell_counts = np.unique(cluster_ids, return_counts=True)
        for i in range(len(clusters)):
            if cell_counts[i] < 50:
                cluster_ids[cluster_ids==clusters[i]] = clusters[0]
                
        db.obs["cluster_id"] = cluster_ids
        print("# of clusters: ", len(np.unique(cluster_ids)))



        ################## Assignment of unclustered cells ####################
        print("Number of unclustered cells: ", db[db.obs.cluster_id == "-1_" + db.uns["name"]].shape[0])
        clustered_cells = db[db.obs.cluster_id != "-1_" + db.uns["name"]].obs.index.tolist()
        KDtree_all = cKDTree(db[clustered_cells].obsm["umap"])
        for i in range(db.shape[0]):
            if cluster_ids[i] == "-1_" + db.uns["name"]:
                closest_i = clustered_cells[KDtree_all.query(db[i].obsm["umap"])[1][0]]
                cluster_ids[i] = db[closest_i].obs.cluster_id.values[0]
        db.obs["cluster_id"] = cluster_ids
        whole_cluster_list.extend(cluster_ids)

        
        ################## Calculate protein importances ######################
        for cluster in db.obs.cluster_id.unique():
            print("HVGs are calculating for ", cluster)
            c_label = [1 if x == cluster else 0 for x in list(db.obs.cluster_id)]
            cluster_HVGs[cluster] = get_HVGs(db, c_label, NIP)
            print(cluster, cluster_HVGs[cluster])
        #######################################################################

    exp_data.whole.obs["cluster_id"] = whole_cluster_list

    # ################## calculate cluster matchings ########################
        
    matched_clusters = []
    unmatched_clusters = []
    diff_map = {}

    cluster_relations = {}
    for cluster in cluster_HVGs: 
        cluster_relations[cluster] = []
        most_diff = None
        cluster_cells = exp_data.whole[exp_data.whole.obs["cluster_id"] == cluster] 
        print(cluster, cluster_cells.shape, np.unique(cluster_cells.obs["cluster_s"], return_counts=True))
        for cluster2 in cluster_HVGs:
            if cluster != cluster2 and cluster.split("_")[1:] != cluster2.split("_")[1:]:

                rank = cluster_distance(cluster_HVGs[cluster], cluster_HVGs[cluster2])
                cluster_relations[cluster].append((rank, cluster2))
                
                print(cluster, cluster2, rank)
#                 if rank > cluster_treshold:
#                     cluster_relations[cluster].append((rank, cluster2))

                if most_diff is None or rank < most_diff[0]:
                    most_diff = (rank, cluster2)
        cluster_relations[cluster] = sorted(cluster_relations[cluster], key=lambda x: x[0],  reverse = True)
        if len(cluster_relations[cluster]) > 1:
            cluster_relations[cluster] = cluster_relations[cluster][:1]
        if len(cluster_relations[cluster]) > 0:
            cluster_relations[cluster].sort(key = lambda x: x[0], reverse = True)
            cluster_relations[cluster] = [x[1] for x in cluster_relations[cluster]]
        diff_map[cluster] = most_diff[1]
    #########################################################################
    
    ################## Calculate connected componenets ######################
    
    batch_relations = {}
    r_batch_relations = {}
    components = {}
    edges = []
    # print(cluster_relations)
    for cluster in cluster_relations:
        for related_cluster in cluster_relations[cluster]:
            # if cluster not in cluster_relations[related_cluster]:
            #     cluster_relations[cluster].remove(related_cluster)
            # else:
            #     edges.append((cluster, related_cluster)) ###################### Open this part if threshold is on use.
            edges.append((cluster, related_cluster))
        if len(cluster_relations[cluster]) == 0:
            unmatched_clusters.append(cluster)

    # print(cluster_relations)
    total_unmatched = 0
    for cluster in unmatched_clusters:
        total_unmatched += exp_data.whole[exp_data.whole.obs["cluster_id"] == cluster].shape[0]
    print("Unmatched Clusters:" , len(unmatched_clusters),"-",len(cluster_HVGs), " (" + str(total_unmatched) + ") ", unmatched_clusters)
    
    DG = nx.Graph()
    DG.add_edges_from(edges)
    g_counter = 0

    for cmp in nx.connected_components(DG):
        new_group = "group_" + str(g_counter)
        batch_relations[new_group] = []
        components[new_group] = list(cmp)
        for cluster in cmp:
            for related_cluster in cluster_relations[cluster]:
                if (related_cluster, cluster) not in batch_relations[new_group]:
                    batch_relations[new_group].append((cluster, related_cluster))
            r_batch_relations[cluster] = new_group
        g_counter +=1
    # for group in batch_relations:
    #     print(batch_relations)
    
    ######################Refine Groups##########################################
    
    # for group in batch_relations:
    #     for cluster in batch_relations[group]:
    #         similarity = 0
    #         for cluster_o in batch_relations[group]:
    #             if cluster != cluster_o:
    #                 rank = cluster_distance(cluster_HVGs[cluster], cluster_HVGs[cluster2])
    #                 if rank > 0.5:
    #                     similarity += 1
    #         if similarity < len(batch_relations[group]) * 0.3:
    #             batch_relations
                    
    
    ################## Second chance for unmatched clusters ######################
#     for group in components:
#         c_label = [1 if x in components[group] else 0 for x in list(exp_data.whole.obs.cluster_id)]
#         cluster_HVGs[group] = get_HVGs(exp_data.whole, c_label, NIP)

#     for cluster in list(unmatched_clusters):
#             most_similar = (None, None)
#             for group in batch_relations:
#                 rank = cluster_distance(cluster_HVGs[cluster], cluster_HVGs[group])
#                 if most_similar[0] is None or most_similar[0] < rank:
#                     most_similar = (rank, group)

#             if most_similar[0] != None and most_similar[0] > cluster_treshold:
#                 batch_relations[most_similar[1]].append((cluster, components[most_similar[1]][0]))
#                 cluster_relations[cluster] = [components[most_similar[1]][0]]
#                 r_batch_relations[cluster] = most_similar[1]
#                 unmatched_clusters.remove(cluster)
#                 # print(cluster, most_similar)

#     tmp_edges = []
#     for cluster in list(unmatched_clusters):
#             tmp_relations = []
#             most_similar = (None, None)
#             for cluster2 in unmatched_clusters:
#                 if cluster != cluster2 and cluster.split("_")[1:] != cluster2.split("_")[1:]:
#                     rank = cluster_distance(cluster_HVGs[cluster], cluster_HVGs[cluster2])
#                     # print(cluster, "-", cluster2, rank)
#                     if rank > cluster_treshold - 0.05:
#                         tmp_relations.append((cluster, cluster2))

#             if len(tmp_relations) > 0:
#                 cluster_relations[cluster]= [x[1] for x in tmp_relations]
#                 tmp_edges.extend(tmp_relations)


#     DG = nx.Graph()
#     DG.add_edges_from(tmp_edges)

#     for cmp in nx.connected_components(DG):
#         new_group = "group_" + str(g_counter)
#         batch_relations[new_group] = []
#         components[new_group] = list(cmp)
#         for cluster in cmp:
#             for related_cluster in cluster_relations[cluster]:
#                 if (related_cluster, cluster) not in batch_relations[new_group]:
#                     batch_relations[new_group].append((cluster, related_cluster))
#             r_batch_relations[cluster] = new_group
#             unmatched_clusters.remove(cluster)
#         g_counter +=1
#         print("new_group ---->", new_group, " : ", components[new_group])

    batch_list = {}
    new_groups = []
    for cluster in unmatched_clusters:
        # batch_idx = cluster.split('_')[1]
        cluster_cells = exp_data.whole[exp_data.whole.obs["cluster_id"] == cluster] 
        print(cluster, cluster_cells.shape, max(set(cluster_cells.obs["cluster_s"]), key = cluster_cells.obs["cluster_s"].tolist().count))
        batch_idx = exp_data.whole[exp_data.whole.obs["cluster_id"] == cluster].obs["batch_id"].tolist()[0]
        if batch_idx not in batch_list:
            new_group = "group_" + str(g_counter)
            new_groups.append(new_group)
            batch_list[batch_idx] = new_group
            components[new_group] = [cluster]
            cluster_relations[cluster] = [cluster]
            batch_relations[new_group] = []
            g_counter += 1
        batch_relations[batch_list[batch_idx]].append((cluster, components[batch_list[batch_idx]][0]))
        components[batch_list[batch_idx]].append(cluster)
        cluster_relations[cluster] = [components[batch_list[batch_idx]][0]]

        r_batch_relations[cluster] = batch_list[batch_idx]

    for group in components:
        print("Batch group -> ", group, components[group])
        

    exp_data.whole.obs["group_id"] = np.array(["no-group" if cell not in r_batch_relations else r_batch_relations[cell] for cell in exp_data.whole.obs["cluster_id"]])

    ################## Cell anchor establish ######################

    cell_mappings = {}
    # cell_mappings_n = {}
    negatives_for_prediction = {}
    target_mappings = {}
    same_ctr = all_ctr = 0
    cell_type_mapping = {}
    type_dict = {}
    cluster_type = {}
    cell_negs = {}
    for group in batch_relations:
        cell_type_mapping[group] = []
        cell_mappings[group] = []
        # cell_mappings_n[group] = []
        negatives_for_prediction[group] = []
        plain_list = list(set([x[0] for x in batch_relations[group]] + [x[1] for x in batch_relations[group]]))
        if len(plain_list) > 1:
            for i in range(len(plain_list)):
                cluster_q = plain_list[i]
                query_hvgs = list(cluster_HVGs[cluster_q].keys())
                if group in new_groups:
                    target_group = [x for j,x in enumerate(plain_list) if j!=i] 
                else:
                    target_group = [x for j,x in enumerate(plain_list) if j!=i and x.split("_")[1:] != cluster_q.split("_")[1:]] 

                query_cells = exp_data.whole[exp_data.whole.obs["cluster_id"] == cluster_q]
                if extend:
                    for target in target_group:
                        common_hvgs = list(set(query_hvgs) & set(list(cluster_HVGs[target].keys())))
                        target_cells = exp_data.whole[exp_data.whole.obs["cluster_id"] == target]
                        # m1, m2 = find_mutual_nn2(query_cells[:, common_hvgs].X, target_cells[:, common_hvgs].X, 20, 20)
                        m1, m2 = find_mutual_nn3(query_cells[:, common_hvgs].X, target_cells[:, common_hvgs].X)
                        
                        if len(m1) > 0:
                            q_cells_i = query_cells.obs.index.to_numpy()[m1]
                            t_cells_i = target_cells.obs.index.to_numpy()[m2]

                            ## for negative samples
                            neg_cells = exp_data.whole[exp_data.whole.obs["cluster_id"] == diff_map[cluster_q]]

                            neg_targets = neg_cells.obs.index.to_numpy()[np.random.choice(neg_cells.shape[0], len(q_cells_i), replace=True)]

                            cell_mappings[group].extend(list(zip(q_cells_i,t_cells_i, neg_targets)))
                else:
                    target_cells = exp_data.whole[exp_data.whole.obs["cluster_id"].isin(target_group)]
                    neg_cells = exp_data.whole[exp_data.whole.obs["cluster_id"] == diff_map[cluster_q]]

                    m1, m2 = find_mutual_nn3(query_cells.X, target_cells.X)

                    if len(m1) > 0:
                        q_cells_i = query_cells.obs.index.to_numpy()[m1]
                        t_cells_i = target_cells.obs.index.to_numpy()[m2]

                        neg_cells = exp_data.whole[exp_data.whole.obs["group_id"] != group]
                        
                        neg_targets = neg_cells.obs.index.to_numpy()[np.random.choice(neg_cells.shape[0], len(q_cells_i), replace=True)]
                        cell_mappings[group].extend(list(zip(q_cells_i,t_cells_i, neg_targets)))

                    neg_targets = neg_cells.obs.index.to_numpy()[np.random.choice(neg_cells.shape[0], query_cells.shape[0] , replace=True)]
                    negatives_for_prediction[group].extend(neg_targets) 
                if "cluster_s" in query_cells.obs:
                    q_type = max(set(query_cells.obs["cluster_s"]), key = query_cells.obs["cluster_s"].tolist().count)
                    t_type = max(set(target_cells.obs["cluster_s"]), key = target_cells.obs["cluster_s"].tolist().count)
                    cell_type_mapping[group].append((q_type, t_type))
                    cluster_type[cluster_q] = q_type
                    if q_type == t_type:
                        same_ctr += 1
                all_ctr += 1
        else:
            cluster_q = plain_list[0]
            query_cells = exp_data.whole[exp_data.whole.obs["cluster_id"] == cluster_q]
            if "cluster_s" in query_cells.obs:
                cluster_type[cluster_q] = max(set(query_cells.obs["cluster_s"]), key = query_cells.obs["cluster_s"].tolist().count)
                cell_type_mapping[group].append((cluster_type[cluster_q], cluster_type[cluster_q]))
            
            q_cells_i = query_cells.obs.index.to_numpy()
            neg_cells = exp_data.whole[exp_data.whole.obs["cluster_id"] == diff_map[cluster_q]]

            neg_targets = neg_cells.obs.index.to_numpy()[np.random.choice(neg_cells.shape[0], len(q_cells_i), replace=True)]
            cell_mappings[group].extend(list(zip(query_cells.obs.index.to_numpy(), query_cells.obs.index.to_numpy(), neg_targets)))

            negatives_for_prediction[group].extend(neg_targets)


    # print("Cluster matching analysis --> # of same = ", same_ctr, "# of all = ", all_ctr)

    # for group in cell_mappings:
    #     for match in cell_mappings[group]:
    #         if exp_data.whole[match[0]].obs.cluster_id.values[0] == exp_data.whole[match[1]].obs.cluster_id.values[0]:
    #             print(exp_data.whole[match[0]].obs.cluster_id.values[0])


    ################### Report some information about the matchings.###########################

    DG_fig = nx.DiGraph()
    color_map = []
    for cluster in cluster_relations:
        for target_c in cluster_relations[cluster]:
            edges.append((cluster, target_c))
    DG_fig.add_edges_from(edges)
    number_map = {}
    cur = 0
    fig = plt.figure(figsize=(50,50))
    if "cluster_s" in exp_data.whole.obs:
        for node in DG_fig.nodes():
            if cluster_type[node] not in number_map:
                number_map[cluster_type[node]] = cur
                cur += 1
            color_map.append(number_map[cluster_type[node]])
        # print(cluster_type)
        # print("*" * 100)
        # for group in cell_type_mapping:
        #     print(*batch_relations[group], sep = "\n")
        #     print(group, cell_type_mapping[group])
    else:
        color_map = [-1] * len(DG_fig.nodes()) 
    nx.draw(DG_fig, with_labels=True, node_color = color_map, cmap = plt.cm.Set1, node_size = 1000)
    plt.savefig("relations.png")


    cell_matching_acc(exp_data.whole, cell_mappings)

    ##################### Batch-effect correction #################################
    args = Namespace(vtype = None,
                     input_size = NIP, 
                     dense_layer_size = NIP * 3, 
                     latent_size = NIP, 
                     dropout = 0.2, 
                     beta = 1, 
                     epochs = 100, 
                     batch_size = 16,
                     save_model = False)

    ##################### Training VAE models #################################

    cc_HVGs = {}
    regression_models = {}
    gene_sets = {}
    ensemble_model = {}
    for group in batch_relations:
        print(group, " is training...")
        c_label = [1 if x == group else 0 for x in list(exp_data.whole.obs.group_id)]
        cc_HVGs[group] = list(get_HVGs(exp_data.whole, c_label, NIP).keys())
        if extend:
            regression_models[group] = {}
            gene_sets[group] = {}
            i = 0
            remaining_genes = np.array([x for x in exp_data.whole.var.index.values if x not in cc_HVGs[group]])
            n_reg_models = int(remaining_genes.shape[0] / NIP) - 2
            _gene_sets = [remaining_genes[np.arange(x * NIP  , ((x+1) * NIP) - 1)] for x in range(n_reg_models)]
            _gene_sets.append(remaining_genes[np.arange(n_reg_models * NIP , ((n_reg_models + 1) * NIP) - 2)])
            for gene_set in _gene_sets:
                regression_models[group]["set_"+ str(i)] = multiple_regression(exp_data.whole[exp_data.whole.obs.group_id == group], cc_HVGs[group], gene_set)
                gene_sets[group]["set_"+ str(i)] = gene_set
                i += 1

        if len(cell_mappings[group]) > 0:
            tuple_cell, tuple_pos, tuple_neg = zip(*cell_mappings[group])
            train1 = exp_data.whole[list(tuple_cell), cc_HVGs[group]].X
            train2 = exp_data.whole[list(tuple_pos), cc_HVGs[group]].X
            train3 = exp_data.whole[list(tuple_neg), cc_HVGs[group]].X
            ensemble_model[group] = VAE(args)
            print(len(train1), len(train2), len(train3))
            ensemble_model[group].train(train1, train2, train3)

    ##################### Generate new embeddings #################################

    new_features = np.zeros(exp_data.whole.X.shape)
    ctr_group = 0
    noise_counter = 0
    for group in cell_mappings:
        # tuple_cell, tuple_pos, tuple_neg = zip(*cell_mappings[group])
        filtered_cells = exp_data.whole[exp_data.whole.obs.group_id == group]
        old_embs = filtered_cells[:, cc_HVGs[group]].X.copy()
        i_cell_idx = exp_data.whole.obs.index.get_indexer(filtered_cells.obs.index.values)
        current_cells = new_features[i_cell_idx].copy()
        if len(cell_mappings[group]) > 0:
            neg_embs = exp_data.whole[negatives_for_prediction[group], cc_HVGs[group]].X.copy()
            new_embs = ensemble_model[group].predict(old_embs, old_embs, neg_embs)

            if extend:
                for gene_set in gene_sets[group]:
                    set_prots = exp_data.whole.var.index.get_indexer(gene_sets[group][gene_set])
                    set_corrected = regression_models[group][gene_set].predict(new_embs) 
                    current_cells[:, set_prots] = set_corrected
        else:
            new_embs = old_embs
        i_imp_prots = exp_data.whole.var.index.get_indexer(cc_HVGs[group])
        current_cells[:, i_imp_prots] = new_embs
        new_features[i_cell_idx] = current_cells 

    exp_data.whole.obsm["SCPRO-HI"] = new_features
    if "cluster_s" in exp_data.whole.obs:
        _adata = exp_data.whole.copy() 
        _adata.X = new_features
        exp_data.whole.obs["SCPRO-HI-preds"] = calc_preds(_adata)
