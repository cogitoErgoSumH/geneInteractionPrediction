import pandas as pd
import numpy as np
import tqdm
import pickle
import os


def get_gene_list_bulk(file_name):
    """
    建立基因symbol到ensembl转换的字典
    file_name = './data/bulk_gene_list.txt'
    """
    import re
    h = {}
    s_ = open(file_name, 'r')  # gene symbol ID list of bulk RNA-seq
    for line_ in s_:
        search_result = re.search(r'^([^\s]+)\s+([^\s]+)', line_)
        h[search_result.group(1).lower()] = search_result.group(2)  # h [gene symbol] = gene ID
    s_.close()
    return h


def get_gene_list(file_name):
    """
    建立基因symbol到一个id转换的字典
    file_name = './data/sc_gene_list.txt'
    """

    import re
    h = {}
    s_ = open(file_name, 'r')  # gene symbol ID list of sc RNA-seq
    for line_ in s_:
        search_result = re.search(r'^([^\s]+)\s+([^\s]+)', line_)
        h[search_result.group(1).lower()] = search_result.group(2)  # h [gene symbol] = gene ID
    s_.close()
    return h


def get_sepration_index(file_name):
    index_list = []
    s_ = open(file_name, 'r')
    for line_ in s_:
        index_list.append(int(line_))
    s_.close()
    return np.array(index_list)


def delete_zero(x_gene_sc, y_gene_sc):
    gene_df=pd.DataFrame({"x_gene_sc":x_gene_sc,"y_gene_sc":y_gene_sc})
    gene_df=gene_df[gene_df["x_gene_sc"]!=-2]
    gene_df=gene_df[gene_df["y_gene_sc"] != -2]
    print(type(gene_df["x_gene_sc"]))
    return np.array(gene_df["x_gene_sc"]),np.array(gene_df["y_gene_sc"])
root_path="D:/data/cnncdata/"
bulk_gene_list_path=root_path+"bulk_gene_list.txt"
sc_gene_list_path=root_path+"sc_gene_list.txt"
mouse_bulk_path=root_path+"mouse_bulk.h5"
rank_total_gene_rpkm_path=root_path+"rank_total_gene_rpkm.h5"
mmukegg_new_new_unique_rand_labelx_path=root_path+"mmukegg_new_new_unique_rand_labelx.txt"
mmukegg_new_new_unique_rand_labelx_num_path=root_path+"mmukegg_new_new_unique_rand_labelx_num.txt"
TEST_NUM = 3057

h_gene_list_bulk = get_gene_list_bulk(bulk_gene_list_path)
h_gene_list = get_gene_list(sc_gene_list_path)

store = pd.HDFStore(mouse_bulk_path)
rpkm_bulk = store['rpkm']
store.close()
store = pd.HDFStore(rank_total_gene_rpkm_path)
rpkm = store['rpkm']
store.close()

gene_pair_label = []
s = open(mmukegg_new_new_unique_rand_labelx_path)
for line in s:
    gene_pair_label.append(line)
gene_pair_index = get_sepration_index(mmukegg_new_new_unique_rand_labelx_num_path)
s.close()
gene_pair_label_array = np.array(gene_pair_label)
save_dir = os.path.join(os.getcwd(), 'data\samples')
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
for i in tqdm.tqdm(range(len(gene_pair_index) - 1)):
    if i > TEST_NUM:
        break
    start_index = gene_pair_index[i]
    end_index = gene_pair_index[i + 1]
    gene_pairs = gene_pair_label_array[start_index:end_index]
    label_list = []
    bulk_pairs = []
    sc_pairs = []
    for gene_pair in gene_pairs:
        x_gene_name, y_gene_name, label = gene_pair.strip().split()
        label = int(label)
        if label == 1 or label == 0:
            label_list.append([x_gene_name, y_gene_name, label])
            x_gene_ensembl, y_gene_ensembl = h_gene_list_bulk[x_gene_name], h_gene_list_bulk[y_gene_name]
            x_gene_bulk = np.log10(rpkm_bulk[x_gene_ensembl] + 0.01).values
            y_gene_bulk = np.log10(rpkm_bulk[y_gene_ensembl] + 0.01).values
            bulk_pairs.append([x_gene_name, y_gene_name, x_gene_bulk, y_gene_bulk])

            x_gene_id, y_gene_id = int(h_gene_list[x_gene_name]), int(h_gene_list[y_gene_name])
            x_gene_sc = np.log10(rpkm[x_gene_id][:-1] + 0.01).values
            y_gene_sc = np.log10(rpkm[y_gene_id][:-1] + 0.01).values
            x_gene_sc,y_gene_sc=delete_zero(x_gene_sc,y_gene_sc)
            print("x_gene_sc的参数类型是：")
            print(type(x_gene_sc))
            print(type(y_gene_sc))
            sc_pairs.append([x_gene_name, y_gene_name, x_gene_sc, y_gene_sc])

    if len(bulk_pairs) > 0 and len(sc_pairs) > 0:

        with open(f'./data/samples2/label_{i}.pkl', 'wb') as f:
            pickle.dump(label_list, f)
        with open(f'./data/samples2/bulk_{i}.pkl', 'wb') as f:
            pickle.dump(bulk_pairs, f)
        with open(f'./data/samples2/sc_{i}.pkl', 'wb') as f:
            pickle.dump(sc_pairs, f)
        #黄加的
        # with open(f'./data/samples2/label_{i}.pkl', 'wb') as f:
        #     pickle.dump(label_list, f)
        # with open(f'./data/samples2/bulk_{i}.pkl', 'wb') as f:
        #     pickle.dump(bulk_pairs, f)
        # with open(f'./data/samples2/sc_{i}.pkl', 'wb') as f:
        #     pickle.dump(sc_pairs, f)

