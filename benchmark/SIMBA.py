import os
import simba as si
import scanpy as sc
import pandas as pd
import anndata as ad

workdir = '/simba/brain'
si.settings.set_workdir(workdir)

si.settings.set_figure_params(dpi=80,
                              style='white',
                              fig_size=[5,5],
                              rc={'image.cmap': 'viridis'})

# make plots prettier
from matplotlib_inline.backend_inline import set_matplotlib_formats
set_matplotlib_formats('retina')

#load example data
#adata_CG = si.datasets.rna_10xpmbc3k()
data_raw=pd.read_csv("Adult-Brain_Lake_dge.txt",sep=",",index_col=0,header=0)
df_t = data_raw.T
var = pd.DataFrame(index=df_t.columns)
obs = pd.DataFrame(index=df_t.index)
adata=ad.AnnData(df_t.values,obs=obs,var=var,dtype='float32')

meta=pd.read_csv("Adult-Brain_Lake_meta_data.csv",sep=",",index_col=0,header=0)
adata.obs['celltype']=meta['assigned_cluster']

#preprocessing
adata_CG=adata
si.pp.filter_genes(adata_CG,min_n_cells=3)
si.pp.cal_qc_rna(adata_CG)

si.pp.filter_cells_rna(adata,min_n_genes=100)
si.pp.normalize(adata_CG,method='lib_size')
si.pp.log_transform(adata_CG)

#discretize RNA expression
si.tl.discretize(adata_CG,n_bins=5)
si.pl.discretize(adata_CG,kde=False)

#generate a graph for training
si.tl.gen_graph(list_CG=[adata_CG],
                layer='simba',
                use_highly_variable=False,
                dirname='graph1')

### PBG training
# modify parameters
dict_config = si.settings.pbg_params.copy()
# dict_config['wd'] = 0.015521
dict_config['wd_interval'] = 10 # we usually set `wd_interval` to 10 for scRNA-seq datasets for a slower but finer training
dict_config['workers'] = 12 #The number of CPUs.

#Plotting training metrics to monitor overfitting and evaluate the final model. Ideally, after a certain number of epochs, the metric curve should stabilize and remain steady.
#si.pl.pbg_metrics(fig_ncol=1)
## start training
si.tl.pbg_train(pbg_params = dict_config, auto_wd=True, save_wd=True, output="model2")

### Post-training analysis
# read in entity embeddings obtained from pbg training.
dict_adata = si.read_embedding()

adata_C = dict_adata['C']  # embeddings of cells
adata_G = dict_adata['G']  # embeddings of genes

gene_emb=pd.DataFrame(adata_G.X,index=adata_G.obs.index)

gene_emb.to_csv("brain_embedding_SIMBA_default.txt")
