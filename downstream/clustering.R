library(Rtsne)
library(data.table)
library(lisi)
library(ggpubr)
library(dplyr)
library(cluster)
library(factoextra)

final_data=fread("brain_geneembedding.txt")
set.seed(321)
tsne_out = Rtsne(
      unique(final_data[,c(1:16)]),
      dims = 2,
      pca = T,
      max_iter = 1000,
      theta = 0.4,
      perplexity = 40,
      verbose = F,
      check_duplicates = FALSE
) 

tsne_result = as.data.frame(tsne_out$Y)
colnames(tsne_result) = c("tSNE1","tSNE2")
tsne_result=cbind(final_data,tsne_result)

#Kmeans clustering
km <- kmeans(tsne_result[,c("tSNE1","tSNE2")], centers = 10, nstart = 25)
input<-data.frame(tsne_result)
cluster<-fviz_cluster(km, data = tsne_result[,1:16])
final_data <- cbind(tsne_result, cluster$data)
final_data$cluster<-as.factor(final_data$cluster)
p<-ggplot(final_data,aes(tSNE1,tSNE2,color=cluster)) +geom_point(alpha=1,size=0.4)+
      #scale_color_manual(values = mypalette)+
      theme(legend.key = element_blank(),panel.background = element_blank(),panel.border = element_rect(color="black",fill = "transparent"))
write.table(final_data,paste0("brain_embedding_clusters.csv"),sep=",",row.names=F)
