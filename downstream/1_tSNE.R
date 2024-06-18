library(Rtsne)
library(data.table)
library(ggplot2)
library(lisi)
library(ggpubr)
library(dplyr)
library(cluster)
library(factoextra)
library(RColorBrewer)
library(randomcoloR)

embedding_clust=function(emb){
    latent<-fread(emb,header=F)
    gene<-fread("./data/node_order.txt",header=F)
    input<-cbind(gene,latent)
    names(input)[1]<-"Symbol"

    label<-fread("../data/HSIAO_HOUSEKEEPING_GENES_405_new.txt",header=F)
    names(label)[1]<-"Symbol"
    label$housekeeping<-1
    final_data<-data.frame(left_join(input,label,by="Symbol"))
    final_data[is.na(final_data$housekeeping),"housekeeping"]<-0
    set.seed(321)
    tsne_out = Rtsne(
      unique(final_data[,c(2:17)]),
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
    tsne_result=cbind(final_data[row.names(unique(final_data[,c(2:17)])),],tsne_result)

    set.seed(1)
    km <- kmeans(tsne_result[,c("tSNE1","tSNE2")], centers = 10, nstart = 25)
    input<-data.frame(tsne_result)
    row.names(tsne_result)<-tsne_result$Symbol
    cluster<-fviz_cluster(km, data = tsne_result[,2:17])
    final_data <- cbind(tsne_result, cluster$data)
    final_data$cluster<-as.factor(final_data$cluster)
    return(final_data)
}

fn_list=list.files("./data",'embedding.txt',full.names=TRUE)
prefix="./result/tsne_perplexity40_"
for(i in 1:length(fn_list)){
  final_data=embedding_clust(fn_list[i])
  p<-ggplot(final_data,aes(tSNE1,tSNE2,color=cluster)) +geom_point(alpha=1,size=0.4)+
      scale_color_manual(values = mypalette)+
      theme(legend.key = element_blank(),panel.background = element_blank(),panel.border = element_rect(color="black",fill = "transparent"))
  assign(paste0("p",i),p)
  ggsave(p,filename = paste0(prefix,"_tsnekmeans10.pdf"),dpi=300)
  write.table(final_data,paste0(prefix,"_tsnekmeans10.csv"),sep=",",row.names=F)

  final_data$housekeeping<-as.factor(final_data$housekeeping)
  tsne_result2=final_data[which(final_data$housekeeping==1), ]
  p<-ggplot(final_data,aes(tSNE1,tSNE2,color=housekeeping)) +geom_point(alpha=1,size=0.4)+
    scale_color_manual(values = c("#F6EDEF","#F69896"))+
    geom_point(data=tsne_result2, mapping=aes(tSNE1, tSNE2), size=0.6, color="red")+
    theme(legend.key = element_blank(),panel.background = element_blank(),panel.border = element_rect(color="black",fill = "transparent"))
  ggsave(p,filename = paste0(prefix,"_HSIAO_HOUSEKEEPING_GENES_405_new.pdf"),dpi=300)
}
