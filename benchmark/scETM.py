import anndata as ad
import pandas as pd
import scanpy as sc
from scETM import scETM, UnsupervisedTrainer, evaluate, prepare_for_transfer
sc.set_figure_params(dpi=120, dpi_save=250, fontsize=10, figsize=(10, 10), facecolor="white")

data_raw=pd.read_csv("Adult-Brain_Lake_dge.txt",sep=",",index_col=0,header=0)
df_t = data_raw.T
var = pd.DataFrame(index=df_t.columns)
obs = pd.DataFrame(index=df_t.index)
adata=ad.AnnData(df_t.values,obs=obs,var=var,dtype='float32')

sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.normalize_total(adata, exclude_highly_expressed=True, target_sum=1e6)
sc.pp.log1p(adata)
sc.pp.scale(adata)
adata.var_names_make_unique()

meta=pd.read_csv("Adult-Brain_Lake_meta_data.csv",sep=",",index_col=0,header=0)
adata.obs['assigned_cluster']=meta['assigned_cluster']

mp=adata

mp_model = scETM(mp.n_vars, mp.obs.batch_indices.nunique())
trainer = UnsupervisedTrainer(mp_model, mp, test_ratio=0.1)

trainer.train(n_epochs = 9000, eval_every = 3000, eval_kwargs = dict(cell_type_col = 'assigned_cluster'), save_model_ckpt = False)

#Get scETM output embeddings
#Evaluate the model and save the embedding plot
#evaluate(mp, embedding_key="delta", plot_fname="scETM_MP", plot_dir="/xtdisk/liufan_group/linshq/work/2022project/cellphe/code/modularity_aware_gae-main/benchmark/scETM/brain/default")

mp_model.get_all_embeddings_and_nll(mp)
gene_emb=pd.DataFrame(mp.varm['rho'],index=mp.var.index)

gene_emb.to_csv("brain_embedding_scETM_epoch9000.txt")
