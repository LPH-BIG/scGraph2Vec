##################### Housekeeping genes ##################

brain<-fread("./result/brain_perplexity40_tsne2d_hier-clust_maxclust127.csv")
tsne_result=brain[,-1]
tsne_result$housekeeping<-as.factor(tsne_result$housekeeping)
tsne_result2=tsne_result[which(tsne_result$housekeeping==1), ]
p<-ggplot(tsne_result,aes(tSNE1,tSNE2,color=housekeeping)) +
	geom_point(alpha=1,size=0.4)+scale_color_manual(values = c("grey","red"))+
	geom_point(data=tsne_result2, mapping=aes(tSNE1, tSNE2), size=0.6, color="red")+
	theme(legend.key = element_blank(),panel.background = element_blank(),panel.border = element_rect(color="black",fill = "transparent"))
ggsave(p,filename = "brain_HSIAO_HOUSEKEEPING_GENES_405_new.pdf",dpi=300)

#################### Topological characteristics #################
library(igraph)
library(viridis)
edgelist=fread("../data/adjacency_edgelist.txt",header=F)
brain=fread("./result/brain_perplexity40_tsne2d_hier-clust_maxclust127.csv")

input=NULL
for(i in 1:max(brain$tsne2d_hier_clust)){
  cluster=brain[brain$tsne2d_hier_clust==i,"Symbol"]
  names(edgelist)=c("Symbol","Symbol2")
  cluster_edge=merge(edgelist,cluster,by="Symbol")
  names(edgelist)=c("Symbol2","Symbol")
  cluster_edge2=merge(edgelist,cluster,by="Symbol")
  cluster_edge=rbind(cluster_edge,cluster_edge2[,c("Symbol","Symbol2")])
  cluster_edge=unique(cluster_edge)

  names(cluster_edge)=c("Symbol2","Symbol")
  with_cluster=merge(cluster_edge,cluster,by="Symbol")
  without_cluster=anti_join(cluster_edge,cluster,by="Symbol")
  with_g <- graph_from_edgelist(as.matrix(with_cluster[,c("Symbol","Symbol2")]),directed = FALSE)
  without_g <- graph_from_edgelist(as.matrix(without_cluster[,c("Symbol","Symbol2")]),directed = FALSE)

  with_g_density=graph.density(with_g)
  without_g_density=graph.density(without_g)

  with_closeness=data.frame(closeness(with_g,normalized = T))
  summary(with_closeness$closeness.with_g..normalized...T.)
  without_closeness=data.frame(closeness(without_g,normalized = T))
  summary(without_closeness$closeness.without_g..normalized...T.)
  input=rbind(input,data.frame(symbol=rownames(with_closeness),closeness=with_closeness$closeness.with_g.,gene_type="with_cluster_genes",graph.density=with_g_density,cluster=paste0("cluster",i),label=i))
  input=rbind(input,data.frame(symbol=rownames(without_closeness),closeness=without_closeness$closeness.without_g.,gene_type="without_cluster_genes",graph.density=without_g_density,cluster=paste0("cluster",i),label=i))
}

mean_closeness=NULL
for(i in unique(input$cluster)){
  tmp=data.frame(cluster=i,gene_type="with_cluster_genes",mean_closeness=mean(input[input$cluster==i & input$gene_type=="with_cluster_genes",]$closeness,na.rm=T))
  tmp=rbind(tmp,data.frame(cluster=i,gene_type="without_cluster_genes",mean_closeness=mean(input[input$cluster==i & input$gene_type=="without_cluster_genes",]$closeness,na.rm=T)))
  mean_closeness=rbind(mean_closeness,tmp)
}
library(ggsci)
p<-ggplot(mean_closeness, aes(gene_type, mean_closeness, fill = gene_type)) + 
  geom_violin()+
  geom_boxplot(width = 0.1) + 
  scale_fill_jco() +
  geom_jitter(shape = 16, size = 1, position = position_jitter(0.2)) + 
  geom_signif(comparisons = comparisons, 
              map_signif_level = F, 
              textsize =3, 
              step_increase = 0.2) + 
  theme_classic()
ggsave(p,filename = "allclusters_closeness.compare.pdf",dpi=300,width=7,height=5)

#################### GO enrichment ###################

brain$tsne2d_hier_clust<-as.numeric(brain$tsne2d_hier_clust)
for(clus in 1:max(brain$tsne2d_hier_clust)){
gene.df <- bitr(brain[brain$tsne2d_hier_clust==clus,]$Symbol,fromType="SYMBOL",toType="ENTREZID", OrgDb = org.Hs.eg.db    
gene <- gene.df$ENTREZID

ego_CC <- enrichGO(gene = gene,
                   OrgDb=org.Hs.eg.db,
                   keyType = "ENTREZID",
                   ont = "CC",
                   pAdjustMethod = "BH",
                   minGSSize = 1,
                   pvalueCutoff = 0.01,
                   qvalueCutoff = 0.05,
                   readable = TRUE)

ego_BP <- enrichGO(gene = gene,
                   OrgDb=org.Hs.eg.db,
                   keyType = "ENTREZID",
                   ont = "BP",
                   pAdjustMethod = "BH",
                   minGSSize = 1,
                   pvalueCutoff = 0.01,
                   qvalueCutoff = 0.05,
                   readable = TRUE)

ego_MF <- enrichGO(gene = gene,
                   OrgDb=org.Hs.eg.db,
                   keyType = "ENTREZID",
                   ont = "MF",
                   pAdjustMethod = "BH",
                   minGSSize = 1,
                   pvalueCutoff = 0.01,
                   qvalueCutoff = 0.05,
                   readable = TRUE)

display_number = c(10, 10, 10)
ego_result_BP <- as.data.frame(ego_BP)[1:display_number[1], ]
ego_result_CC <- as.data.frame(ego_CC)[1:display_number[2], ]
ego_result_MF <- as.data.frame(ego_MF)[1:display_number[3], ]

go_enrich_df <- data.frame(
ID=c(ego_result_BP$ID, ego_result_CC$ID, ego_result_MF$ID),Description=c(ego_result_BP$Description,ego_result_CC$Description,ego_result_MF$Description),
GeneNumber=c(ego_result_BP$Count, ego_result_CC$Count, ego_result_MF$Count),
type=factor(c(rep("biological process", display_number[1]), 
              rep("cellular component", display_number[2]),
              rep("molecular function", display_number[3])), 
              levels=c("biological process", "cellular component","molecular function" )))

for(i in 1:nrow(go_enrich_df)){
  description_splite=strsplit(go_enrich_df$Description[i],split = " ")
  description_collapse=paste(description_splite[[1]][1:5],collapse = " ") 
  go_enrich_df$Description[i]=description_collapse
  go_enrich_df$Description=gsub(pattern = "NA","",go_enrich_df$Description)
}

go_enrich_df$type_order=factor(rev(as.integer(rownames(go_enrich_df))),labels=rev(go_enrich_df$Description))
COLS <- c("#66C3A5", "#8DA1CB", "#FD8D62")

p<-ggplot(data=go_enrich_df, aes(x=type_order,y=GeneNumber, fill=type)) +
  geom_bar(stat="identity", width=0.8) +
  scale_fill_manual(values = COLS) +
  coord_flip() +
  xlab("GO term") + 
  ylab("Gene_Number") + 
  labs(title = "The Most Enriched GO Terms")+
  theme_bw()
ggsave(p,filename = paste0("cluster",clus,"_brain_GO.pdf"),dpi=300,height=10,width=5)
}

################### Cell-type specificity: GSEA analysis ###############
library(ggplot2)
library(clusterProfiler)
library(org.Hs.eg.db)
library(enrichplot)

pbmc=loadRDS("Adult-Brain_Lake_seurat-object.rds")
Idents(pbmc)=pbmc@meta.data$celltype
markers=FindAllMarkers(pbmc,only.pos=T,min.pct=0,logfc.threshold=0)

gmts=brain[,c("tsne2d_hier_clust","Symbol")]
names(gmts)<-c("term","gene")
gmts$term=paste0("cluster",gmts$term)
for(celltype in unique(markers$cluster)){
  submarkers <- markers[markers$cluster==celltype,]
  geneList=submarkers$avg_log2FC
  names(geneList)=submarkers$gene
  geneList=sort(geneList,decreasing = T)

  library(GSEABase)
  egmt <- GSEA(geneList, TERM2GENE=gmts, eps=0, verbose=TRUE)
  subresult=egmt@result
  if(nrow(subresult)==0){
    next
  }else{
  subresult$celltype=celltype
  result=rbind(result,subresult)
  png(paste0("gsea_brain_cluster",celltype,".png"))
  gseaplot2(egmt, 1:nrow(subresult), pvalue_table = TRUE)
  dev.off()
}
}
fwrite(result,"./result/gsea_brain_allcluster_and_allcelltype_results.csv")

library(randomcoloR)
palette <- randomColor(count = 60)
mypalette <- distinctColorPalette(60)
result$ID <- factor(result$ID,levels=c("cluster5","cluster8","cluster9","cluster11","cluster13","cluster21","cluster24","cluster26","cluster33","cluster36","cluster50","cluster53","cluster55","cluster59","cluster64","cluster66","cluster69","cluster75","cluster80","cluster83","cluster84","cluster88","cluster89","cluster91","cluster96","cluster104","cluster107","cluster108","cluster112","cluster113","cluster114","cluster115","cluster117","cluster118","cluster124","cluster126","cluster127"))
p=ggplot(data = result) + 
  geom_point(mapping = aes(x = ID, y = -log10(p.adjust), color = celltype))+
  scale_color_manual(values = mypalette)+
    theme(legend.key = element_blank(),panel.background = element_blank(),panel.border = element_rect(color="black",fill = "transparent"),axis.text.x = element_text(angle=60, hjust=1, vjust=1))
ggsave(p,filename = "gsea_brain_allcluster_and_allcelltype_results.pdf",dpi=300,height=5,width=10)

#heatmap of NES values
p=ggplot(result,aes(x=Description,y=celltype))+
  geom_tile(aes(fill=NES), colour = "black")+
  scale_fill_gradientn(colours = rev(brewer.pal(11, "Spectral"))) +
  theme_minimal()+xlab(NULL) + ylab(NULL) +
  theme(panel.background = element_rect(color="black"),
        axis.text.x = element_text(angle = 90,hjust = 0,vjust= 0.5))
ggsave(p,filename = "gsea_brain_allcluster_and_allcelltype_results_NES-heatmap.pdf",dpi=300,height=3,width=6)


################### Tissue specificity ##################
alltrait=fread("./data/alltissue_clusters.csv")

result=NULL
index=which(names(alltrait)=="brain_tsne2d_hier_clust")
other_trait=names(alltrait)[-index]
for(phe in other_trait[-1]){
  input=alltrait[,c("Symbol",names(alltrait)[index],phe)]
  phename=strsplit(names(input)[3],"_")[[1]][1]
  names(input)<-c("Symbol","tsne2d_hier_clust1","tsne2d_hier_clust2")
  input<-input[complete.cases(input),]
  res = NULL
  for(i in 1:max(input$tsne2d_hier_clust1)){
    fisher_res = NULL
    for(j in 1:max(input$tsne2d_hier_clust2)){
      N=nrow(input)
      M=nrow(input[input$tsne2d_hier_clust2==j,])

      n=nrow(input[input$tsne2d_hier_clust1==i,])
      k=nrow(input[input$tsne2d_hier_clust1==i & input$tsne2d_hier_clust2==j,])
      d <- data.frame(gene.not.interest=c(M-k, N-M-n+k), gene.in.interest=c(k, n-k))
      row.names(d) <- c("In_category", "not_in_category")
      fisher_res=append(fisher_res,fisher.test(d)$p.value)
    }
    enrich=data.frame(t(fisher_res))
    res=data.frame(rbind(res,enrich))
    rownames(res)[nrow(res)]=paste0("brain_cluster",i) 
  }
  names(res)<-paste0(phename,"_cluster",seq(1,max(input$tsne2d_hier_clust2)))
  if(is.null(result)){
    result=res
  }else{
    result<-cbind(result,res)
  }
}
if(min(result)==0){
  result[result==0]=10^-(-log10(min(result[result!=0]))+2)
}
fwrite(result,"./result/brain_vs_othertissue_overlaptest.csv",row.names=T)

#clusters with less than 5% cluster overlap with other tissues are selected as tissue-specific clusters.
index=NULL
num=NULL
for(i in 1:nrow(result)){
  if(sum(result[i,]<0.05)<ncol(result)*0.05){
    index=append(index,i)
    num=append(num,sum(result[i,]<0.05))
  }
}

#Draw heatmap of tissue-specific clusters
sub=data.frame(result[index,])
rownames(sub)=sub$V1
sub=sub[,-1]
pdf("brain_vs_othertissue_overlap_only_specific_clusters.pdf",width=7,height=4)
pheatmap(t(-log10(sub)),
         show_rownames = FALSE,
         show_colnames = TRUE,
         name="-log10(P)",
         cluster_rows = TRUE,
         cluster_cols = TRUE,
         clustering_method="ward.D",
         color=c1
)
dev.off()

