scvae train scVAE/Adult-Brain_Lake_dge_normalized.tsv -m VAE --split-data-set --splitting-fraction 0.95 -r negative_binomial -l 16 -H 256 64 16 -w 200 -e 400 --learning-rate 0.0001

scvae train /scVAE/Adult-Brain_Lake_dge_normalized.tsv -m GMVAE -l 16 -H 256 64 16 -w 200 -e 100 --learning-rate 0.0001
