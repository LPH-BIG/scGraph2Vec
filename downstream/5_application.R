################ COVID-19 ###################
fc<-fread("./data/apply/COVID-19/covid19_KatherineA.Overmyer_2021_tableS1.csv")
gene<-fread("./data/apply/COVID-19/lung_tsne2d_hier-clust_maxclust120_gwasseed-clust-eps0.5pts6.csv")
step1=fread("./data/apply/COVID-19/step1.csv")
magma<-fread("./data/apply/COVID-19/COVID19_HGI_A2_ALL_eur_leave23andme_20220403_GRCh37.tsv.gz.genes.out")

names(fc)[2]<-"Symbol"
res<-merge(fc,gene[,c("Symbol","gwasseed_clust")],by="Symbol")
res[res$gwasseed_clust!=-1,"gwasseed_clust"]<-1
res$gwasseed_clust<-as.factor(res$gwasseed_clust)

## Verify the significant DEGs in case and control identified by gwas magma
tmp=res
tmp$marker="NotSig"
tmp[tmp$`Log2(COVID-19/non-COVID-19)`<c(-1)&tmp$magma_gene==TRUE,"marker"]<-"Down"
tmp[tmp$`Log2(COVID-19/non-COVID-19)`>1&tmp$magma_gene==TRUE,"marker"]<-"Up"
tmp$label=ifelse(tmp$marker == "Down" | tmp$marker == "Up", as.character(tmp$Symbol), '')
tmp$marker=as.factor(tmp$marker)
p<-ggplot(tmp,aes(tmp$`Log2(COVID-19/non-COVID-19)`,-log10(`P`))) +
  geom_point(aes(color = marker)) + 
  labs(title="volcanoplot", 
       x="Log2(COVID-19/non-COVID-19)", 
       y="-log10(MAGMA_PValue)") + 
  scale_color_manual(values = c("#546de5", "#d2dae2","#ff4757")) +
  geom_hline(yintercept=c(-log10(0.05/nrow(tmp))),linetype=2)+ 
  geom_vline(xintercept=c(-1,1),linetype=2)+ 
  geom_text_repel(aes(x = tmp$`Log2(COVID-19/non-COVID-19)`,
                      y = -log10(`P`),          
                      label=label),                       
                  max.overlaps = 10000,
                  size=3,   
                  box.padding=unit(0.5,'lines'),  
                  point.padding=unit(0.1, 'lines'), 
                  segment.color='black', 
                  show.legend=FALSE)+
  theme(legend.key = element_blank(),panel.background = element_blank(),
    panel.border = element_rect(color="black",fill = "transparent"))
ggsave(p,filename = "covid19_magmagenes_volcanoplot.pdf",dpi=300)

## Verify the significant DEGs in case and control identified by seed search
tmp=res[res$gwasseed_clust==1,]
tmp=merge(tmp,step1,by="Symbol")
tmp$marker="NotSig"
tmp[tmp$`Log2(COVID-19/non-COVID-19)`<c(-1)&tmp$`q_value_COVID-19`<0.05,"marker"]<-"Down"
tmp[tmp$`Log2(COVID-19/non-COVID-19)`>1&tmp$`q_value_COVID-19`<0.05,"marker"]<-"Up"
tmp$label=ifelse(tmp$marker == "Down" | tmp$marker == "Up", as.character(tmp$Symbol), '')
tmp$marker=as.factor(tmp$marker)
p<-ggplot(tmp,aes(tmp$`Log2(COVID-19/non-COVID-19)`,-log10(`q_value_COVID-19`))) +
  geom_point(aes(color = marker)) + 
  labs(title="volcanoplot", 
       x="Log2(COVID-19/non-COVID-19)", 
       y="-log10(PValue)") + 
  scale_color_manual(values = c("#546de5", "#d2dae2","#ff4757")) +
  geom_hline(yintercept=-log10(0.05),linetype=2)+
  geom_vline(xintercept=c(-1,1),linetype=2)+
  geom_text_repel(aes(x = tmp$`Log2(COVID-19/non-COVID-19)`,
                      y = -log10(`q_value_COVID-19`),          
                      label=label),
                  max.overlaps = 10000,
                  size=3,
                  box.padding=unit(0.5,'lines'),
                  point.padding=unit(0.1, 'lines'), 
                  segment.color='black',
                  show.legend=FALSE)+
  theme(legend.key = element_blank(),panel.background = element_blank(),
    panel.border = element_rect(color="black",fill = "transparent"))
ggsave(p,filename = "covid19_markergenes-step1_volcanoplot.pdf",dpi=300)

# Heatmap using bulk RNA-seq
tpm<-fread("./data/apply/COVID-19/GSE157103_genes.tpm.tsv.gz")
names(tpm)[1]<-"Symbol"
tmp$tmpmarker=FALSE
tmp[(tmp$`Log2(COVID-19/non-COVID-19)`>1|tmp$`Log2(COVID-19/non-COVID-19)`<c(-1))&tmp$`q_value_COVID-19`<0.05,"tmpmarker"]=TRUE
rna<-merge(tmp[,c("Symbol","tmpmarker")],tpm,by="Symbol")

zscore<-function(x){
  for(i in 1:ncol(x)){
    x[,i] = (x[,i]- mean(x[,i]))/sd(x[,i])
  }
  return(x)
}
rna_z<-zscore(data.frame(t(rna[,-c(1,2)])))
rna_z<-t(rna_z)
rna_z<-rna_z[rna$tmpmarker==TRUE,]
rownames(rna_z) <- rna[rna$tmpmarker==TRUE,]$Symbol

col=colorRampPalette(c("navyblue", "white", "red"))(10)
saminfo<-data.frame(fread("./data/apply/COVID-19/sample_info.csv"))
saminfo<-merge(saminfo,cons,by="sample")
rownames(saminfo) <- saminfo$sample

col_anno <- HeatmapAnnotation(
COVID19=saminfo$COVID.19,
HFD_45=saminfo$HFD_45
)

pdf("covid19_healthylung_markergenes_pheatmap.pdf",height=5,width=10)
ComplexHeatmap::Heatmap(rna_z,
        name = "Zscore",
        show_row_names = F,
        clustering_method_columns="ward.D",
        top_annotation = col_anno,
        column_split = saminfo[,3],
        column_title = NULL,
        show_column_names = F)
dev.off()

################## AD ##############
step1<-fread("./data/apply/AD/step1.csv")
fc<-fread("./data/apply/AD/AD_JamalBWilliams_2021_tables1.csv")
gene<-fread("./result/brain_embedding_clusters_maxclust127_gwasseed-clust-eps0.5pts6.csv")#brain
magma<-fread("./data/apply/AD/Alzheimers_Jansen_2018.txt.gz.genes.out")

################ LUAD ###############
## expression
load("./data/apply/LUAD/dataPrep1_LUAD_TP_TN.RData")
luad=fread("./data/apply/LUAD/tsne2d_hier-clust_maxclust150_LUAD_TCGA_seed_genes.csv")
names(luad)[3]="SYMBOL"
luad=luad[,-1]

luad_tpm=assay(dataPrep1,i="tpm_unstrand")
luad_tpm=ensemblID_to_genesymbol(luad_tpm,list)

for(i in 1:nrow(luad_tpm)){
     luad_tpm[i,which(luad_tpm[i,]==0)]<-min(luad_tpm[i,][which(luad_tpm[i,]!=0)])
}

luad_tpm=log2(luad_tpm)
luad_exp=luad_tpm[luad[luad$cancerseed_clust==12,]$SYMBOL,]

oncogene=luad[luad$cancerseed_clust==i & luad$cancer_genes==TRUE,]$SYMBOL
luad_exp=luad_tpm[luad[luad$cancerseed_clust==i,]$SYMBOL,]

group_list <- ifelse(as.numeric(str_sub(colnames(dataPrep1),14,15))<10,"tumor","normal")
normal_samples <- which(group_list == "normal")
tumor_samples <- which(group_list == "tumor")
normal_gene_expression <- luad_exp[, normal_samples]
tumor_gene_expression <- luad_exp[, tumor_samples]
luad_plot_data=NULL
for(i in 1:nrow(luad_exp)){
    # Extract the subset of the gene expression matrix containing only normal lung samples
    sub_plot_data=rbind(data.frame(symbol=rownames(normal_gene_expression[i,]),expression=t(normal_gene_expression[i,])[,1],sample="normal"),data.frame(symbol=rownames(tumor_gene_expression[i,]),expression=t(tumor_gene_expression[i,])[,1],sample="tumor"))
    luad_plot_data=rbind(luad_plot_data,sub_plot_data)
}
luad_plot_data$sample=as.factor(luad_plot_data$sample)

p <- ggplot(luad_plot_data, aes(x = symbol, y = expression))+
  geom_boxplot(outlier.size = 0.3, aes(fill=factor(sample)),
               position = position_dodge(0.8),size=0.4,lwd=0.4) +  
  ggpubr::stat_compare_means(aes(group = sample),method="wilcox.test", label = "p.signif",color = "black",label.x = 1) +
  guides(fill=guide_legend(title="Sample"))+
  ylab("Log2(TPM)")+
  xlab("Gene")+
  theme_minimal()+
  theme(axis.title=element_text(size=13,face="plain",color="black"),
        axis.text = element_text(size=11,face="plain",color="black"),
        axis.text.x = element_text(angle = 30, hjust = 1, vjust = 1),
        panel.background=element_rect(colour="black",fill=NA),
        panel.grid.minor=element_blank(),
        legend.position="right",
        legend.background=element_rect(colour=NA,fill=NA),
        axis.ticks=element_line(colour="black"))+
  scale_fill_brewer(palette = "Accent")
ggsave(paste0("LUAD_",oncogene,"_neighbor_genes_TCGA_LUAD_log2TPM_expression.pdf"),p,height=5,width=8)
}

### survival analysis
clinical <- GDCquery_clinic(project=c("TCGA-LUAD"),type="clinical")
clinical_trait <- clinical[,c('submitter_id','gender','days_to_death','vital_status','days_to_last_follow_up','ajcc_pathologic_stage')]
clinical_trait$tumor_stage <- as.character(clinical_trait$ajcc_pathologic_stage)
clinical_trait <- clinical_trait[!duplicated(clinical_trait$submitter_id),]
clinical_trait[is.na(clinical_trait$days_to_death),'days_to_death'] <- clinical_trait[is.na(clinical_trait$days_to_death),'days_to_last_follow_up']
clinical_trait <- clinical_trait[,-c(5,6)]
names(clinical_trait) <-c("submitter_id","gender","overall_survival","censoring_status","tumor_stage")
clinical_trait <- clinical_trait %>% dplyr::filter(!(is.na(overall_survival))) %>% dplyr::filter(tumor_stage != 'not reported')
survival_cancer <- clinical_trait

#univariate COX regression analysis
uni_cox_in_bulk <- function(gene_list,survival_info_df){
	library(survival)
	gene_list <- gsub(gene_list,pattern='-',replacement='_')
	gene_list=gene_list[-which(gene_list=="LRRC24")]
	uni_cox <- function(single_gene){
		formula <- as.formula(paste0('Surv(overall_survival,censoring_status)~',single_gene))
		surv_uni_cox <- summary(coxph(formula,data=survival_cancer))
		ph_hypothesis_p <- cox.zph(coxph(formula,data=survival_cancer))$table[1,3]
		if(surv_uni_cox$coefficients[,5]<0.05 & ph_hypothesis_p>0.05){
			single_cox_report <- data.frame('uni_cox_sig_genes'=single_gene,
				'beta'=surv_uni_cox$coefficients[,1],
				'Hazard_Ratio'=surv_uni_cox$coefficients[,-1],
				'z_pvalue'=surv_uni_cox$coefficients[,5],
				'Wald_pvalue'=as.numeric(surv_uni_cox$waldtest[3]),
				'Likelihood_pvalue'=as.numeric(surv_uni_cox$logtest[3]))
			single_cox_report[1,]
		}
	}
	uni_cox_list=NULL
	for(gene in unique(gene_list)){
		uni_cox_list=rbind(uni_cox_list,uni_cox(gene))
	}
	uni_cox_list
}
uni_cox_df <- uni_cox_in_bulk(gene_list=luad[luad$cancerseed_clust!=-1,]$SYMBOL,survival_info_df=survival_cancer)

cutoff=survival_cancer[,gene]<=median(survival_cancer[,gene])
fit <- survfit(Surv(overall_survival, censoring_status) ~cutoff, data = survival_cancer)
pValue = surv_pvalue(fit)$pval
outResult=rbind(outResult,cbind(gene=gene,pvalue=pValue))

res_cox<-coxph(Surv(overall_survival, censoring_status) ~cutoff, data = survival_cancer)

pdf(file=paste(gene,".survival.pdf",sep=""),width=6,height=6)
p <- ggsurvplot(fit,
	pval = T, pval.method = F, pval.coord= c(0.05, 0.05), pval.size = 5,
	conf.int = T,
	risk.table = T, risk.table.col="strata",
	surv.median.line = "hv",
	legend = c(0.8, 0.9),
	legend.title = gene,legend.labs = c("High", "Low"),font.legend = 14,
	font.main = c(16, "bold", "darkblue"),font.tickslab = 12,
	ylab= "Overall Survival",xlab = "Time (days)",font.x = 16, font.y =16, 
	ggtheme = theme_survminer(),
	palette = c("red", "cyan3")
)
p2 <- p$plot + ggplot2::annotate("text",x = 600 , y = 0.15,size=5,
	label = paste("HR :",round(summary(res_cox)$conf.int[1],2))) + ggplot2::annotate("text",x = 1200, y = 0.10,size=5,
	label = paste("(","95%CI:",round(summary(res_cox)$conf.int[3],2),"-",round(summary(res_cox)$conf.int[4],2),")",sep = ""))
print(p,newpage = F)
print(gene)
assign(paste0("p",i),p2)
dev.off()

