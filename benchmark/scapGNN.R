library(scapGNN)
library(data.table)
require(coop)
require(reticulate)
require(parallel)

ConNetGNN<-function(Prep_data,python.path=NULL,miniconda.path = NULL,AE.epochs=1000,AE.learning.rate=0.001,AE.reg.alpha=0.5,use.VGAE=TRUE,
                   GAE.epochs = 300,GAE.learning.rate = 0.01,GAE_val_ratio=0.05,parallel=FALSE,seed=125,GPU.use=FALSE,verbose=TRUE){
  if(!isLoaded("reticulate")){
    stop("The package reticulate is not available!")
  }

  if(parallel){
    if(!isLoaded("parallel")){
      stop("The package parallel is not available!")
    }
  }

  GAE_function<-NULL
  AE_function<-NULL


  ####python
  if(is.null(python.path)){
    condav<-conda_version()
    if(is.character(condav)==TRUE){
      cat(paste(condav,"is available!  \n",sep = " "))

      conda_env<-conda_list()
      p<-which(conda_env[,1]=="scapGNN_env")
      if(length(p)>0){
        cat("The environment scapGNN_env already exists!  \n")
        python.path<-conda_env[p,2]
      }else{
        #creat scapGNN_env
        cat("The environment scapGNN_env not found, will be created!  \n")
        create_scapGNN_env()
        conda_env<-conda_list()
        p2<-which(conda_env[,1]=="scapGNN_env")
        python.path<-conda_env[p2,2]
      }
    }else{
      #install conda
      cat("No conda or miniconda detected, miniconda will be created through the reticulate R package!  \n")
      if (is.null(miniconda.path)) {
        miniconda.path <- reticulate::miniconda_path()
      }
      status <- tryCatch(
        reticulate::install_miniconda(path = miniconda.path),
        error = function(e) {
          return(TRUE)
        }
      )
      if (isTRUE(status)) {
        stop(
          "Error during the installation of miniconda. Please see the website of the ",
          "miniconda for more details",
          call. = FALSE
        )
      }
      cat("Create the environment scapGNN_env!  \n")
      create_scapGNN_env()
      conda_env<-conda_list()
      p2<-which(conda_env[,1]=="scapGNN_env")
      python.path<-conda_env[p2,2]
    }
  }

  if(!py_available()){
    if(python.path=="default"){
      python.path <- Sys.which("python")
    }
    use_python(python.path)
  }


  if(!py_module_available("torch")){
    instPyModule("pytorch")
  }
  if(!py_module_available("sklearn")){
    instPyModule("sklearn")
  }
  if(!py_module_available("scipy")){
    instPyModule("scipy")
  }


  HVexp<-as.matrix(Prep_data[[1]])
  row.names(HVexp)<-NULL
  colnames(HVexp)<-NULL

  cell_features<-as.matrix(Prep_data[[2]])
  row.names(cell_features)<-NULL
  colnames(cell_features)<-NULL

  gene_features<-as.matrix(Prep_data[[3]])
  row.names(gene_features)<-NULL
  colnames(gene_features)<-NULL

  LTMG<-as.matrix(Prep_data[[4]])
  row.names(LTMG)<-NULL
  colnames(LTMG)<-NULL


  cell_adj<-as.matrix(Prep_data[[5]])
  row.names(cell_adj)<-NULL
  colnames(cell_adj)<-NULL

  gene_adj<-as.matrix(Prep_data[[6]])
  row.names(gene_adj)<-NULL
  colnames(gene_adj)<-NULL

  orig_adj<-list(cell_adj,gene_adj)


  if (verbose) {
    cat("Run AutoEncoder  \n")
  }

  if(GPU.use==TRUE){
        source_python(system.file("python", "AutoEncoder_GPU.py", package = "scapGNN"))
  }else{
        source_python(system.file("python", "AutoEncoder.py", package = "scapGNN"))
  }

  if((length(which(HVexp==0))/length(HVexp))>0.8){
    AE.learning.rate <- 0.0001
    GAE.learning.rate <- 0.001
    AE.epochs <- 2000
    GAE.epochs <- 1000
  }

  AE_data<-AE_function(cell_features=cell_features,
                 gene_features=gene_features,
                 exp=HVexp,ltmg_m=LTMG,DNN_epochs=AE.epochs,
                 DNN_learning_rate=AE.learning.rate,
                 reg_alpha=AE.reg.alpha,seed=seed,verbose=verbose)

  if (verbose) {
    cat(paste("Minimum loss:",AE_data[[4]],"\n",sep = " "))
    cat(paste("Minimum loss - epoch:",AE_data[[5]],"\n",sep = " "))
  }

  if (verbose) {
    cat("Run Graph AutoEncoder  \n")
  }

  if(parallel==TRUE){
    ncores<-2
    cl <- makeCluster(ncores)
    clusterEvalQ(cl,library(reticulate))
    clusterEvalQ(cl,library(scapGNN))
    GAE_data<-parLapply(cl,1:2,function(i,AE_data,orig_adj,use.VGAE,GAE.epochs,GAE.learning.rate,seed,python.path,GAE_val_ratio){
      use_python(python.path)

          if(GPU.use==TRUE){
                source_python(system.file("python", "GraphAutoEncoder_GPU.py", package = "scapGNN"))
          }else{
                source_python(system.file("python", "GraphAutoEncoder.py", package = "scapGNN"))
          }

      res<-GAE_function(net_m=orig_adj[[i]],feature_m=AE_data[[i+1]],
                        use_model=use.VGAE,GAE_epochs=GAE.epochs,
                        GAE_learning_rate=GAE.learning.rate,seed=seed,
                        ratio_val=GAE_val_ratio,verbose=verbose)
      if (verbose) {
        cat(paste("Minimum loss:",res[[2]],"\n",sep = " "))
        cat(paste("Minimum loss - epoch:",res[[3]],"\n",sep = " "))
      }

      return(res)
    },AE_data,orig_adj,use.VGAE,GAE.epochs,GAE.learning.rate,seed,python.path,GAE_val_ratio)
    stopCluster(cl)
  }else{
        if(GPU.use==TRUE){
                source_python(system.file("python", "GraphAutoEncoder_GPU.py", package = "scapGNN"))
        }else{
                source_python(system.file("python", "GraphAutoEncoder.py", package = "scapGNN"))
        }

    if (verbose) {
      cat("Construct cell-cell association network  \n")
    }
    cell_gae<-GAE_function(net_m=orig_adj[[1]],feature_m=AE_data[[2]],
                           use_model=use.VGAE,GAE_epochs=GAE.epochs,
                           GAE_learning_rate=GAE.learning.rate,seed=seed,
                           ratio_val=GAE_val_ratio,verbose=verbose)
    if (verbose) {
      cat(paste("Minimum loss:",cell_gae[[2]],"\n",sep = " "))
      cat(paste("Minimum loss - epoch:",cell_gae[[3]],"\n",sep = " "))
    }

    if (verbose) {
      cat("Construct gene-gene association network \n")
    }
    gene_gae<-GAE_function(net_m=orig_adj[[2]],feature_m=AE_data[[3]],
                           use_model=use.VGAE,GAE_epochs=GAE.epochs,
                           GAE_learning_rate=GAE.learning.rate,seed=seed,
                           ratio_val=GAE_val_ratio,verbose=verbose)
    if (verbose) {
      cat(paste("Minimum loss:",gene_gae[[2]],"\n",sep = " "))
      cat(paste("Minimum loss - epoch:",gene_gae[[3]],"\n",sep = " "))
    }

    GAE_data<-list(cell_gae,gene_gae)
  }

  cell_net<-GAE_data[[1]][[1]]
  gene_net<-GAE_data[[2]][[1]]
  gene_emb<-GAE_data[[2]][[4]]
  cg_net<-AE_data[[1]]

  colnames(cg_net)<-colnames(Prep_data[[1]])
  row.names(cg_net)<-row.names(Prep_data[[1]])

  colnames(cell_net)<-colnames(Prep_data[[1]])
  row.names(cell_net)<-colnames(Prep_data[[1]])

  colnames(gene_net)<-row.names(Prep_data[[1]])
  row.names(gene_net)<-row.names(Prep_data[[1]])

  row.names(gene_emb)<-row.names(Prep_data[[1]])

  if (verbose) {
    cat("Done  \n")
  }

  results<-list(cell_network=cell_net,gene_network=gene_net,gene_cell_network=cg_net,gene_emb=gene_emb)
  return(results)
}


data=fread("Adult-Brain_Lake_dge.txt")
data=data.frame(data)
rownames(data)=data$V1
data=data[,-1]

Prep_data <- Preprocessing(data)
ConNetGNN_data <- ConNetGNN(Prep_data,python.path="/software/conda/envs/scvi-env/bin/python")

fwrite(ConNetGNN_data$gene_emb,"brain_scapGNN_gene_embedding.csv",row.names=TRUE)