driver_v2=function(iter, y, x, data, test_famids){

#input variables iter = number of iterations you want of the loop; y = target variable; x = list of features (ivs); 
  #data = training setdata; test_famids = list of family ids for subs in test set in 10 iterations
  
#e.g., driver_v2(1:5, "neoC", brain.region$desikanCA, data, test_famids)
  
#set number of folds
k <- 5

##########DO TRAIN/TEST SPLIT######################

tdata <- data[(data$Family_ID %in% test_famids[[iter]]), ]
data <- data[!(data$Family_ID %in% test_famids[[iter]]), ]

#make data containers
n <- length(data$Subject)
nt <-length(tdata$Subject)
nparamfolds <- 20
nv <- floor(n/k)
cvfits <- array(k,1)
Rsqs <- array(0, dim=c(k,1))

#set hyperparameter details
lammin=.01;
nlam=100;
lamgrid=4^seq(2,-4,length=500)

#split test/train NOT ACCOUNTING FOR TWINS
#test_size <- round(length(data$Subject)*test_percent)
#test_subs <- sample(data$Subject, size=test_size, replace = FALSE, prob = NULL)
#tdata <- data[(data$Subject %in% test_subs), ]
#data <- data[!(data$Subject %in% test_subs), ]

#standardize data
data[c(3:302,363:524)] <- scale(data[c(3:302,363:524)])
tdata[c(3:302,363:524)] <- scale(tdata[c(3:302,363:524)])

#winsorize data
data[c(3:302,363:524)] <- apply(data[c(3:302,363:524)], 2, Winsorize, probs = c(0.05, 0.95), type=7)
tdata[c(3:302,363:524)] <- apply(tdata[c(3:302,363:524)], 2, Winsorize, probs = c(0.05, 0.95), type=7)

#######set up internal CV folds#########
#pick which subjects go into CV
data$cv <- createFolds(data$Subject, k = 5, list = FALSE, returnTrain = FALSE)
cvfits <- list()

for (i in 1:k){
  #make train/val datasets
  traindata <- data[ data$cv!=i,]
  trainID <- traindata$Subject
  valdata  <- data[ data$cv==i,]
  valID <- valdata$Subject
  
  #make design matrices for anlaysis
  design <- as.matrix(traindata[c(x)])
  valdesign <- as.matrix(valdata[c(x)])
  target <- as.matrix(traindata[c(y)])
  valtarget <- as.matrix(valdata[c(y)])
  
  #Fit with elastic net
  cvfit <- fit_enet(data, design, target, nparamfolds, lamgrid)
  cvfits[[i]] <- cvfit;

  #Predict on validation set
  Rsq  <- fit_test(cvfit, valdesign, valtarget, k)
  Rsqs[[i]] = Rsq
}

#set up design matrices for running on test set
design <- as.matrix(data[c(x)])
tdesign <- as.matrix(tdata[c(x)])
target <- as.matrix(data[c(y)])
ttarget <- as.matrix(tdata[c(y)])

#pick best model from k-fold CV
maxRsq <- max(Rsqs)
meanRsq <- mean(Rsqs)
bestfold <- which.max(Rsqs)
cvfit <- cvfits[[bestfold]]

#test best model on test set
Test_Rsq <- fit_test(cvfit, tdesign, ttarget, k)

#get betas from best model
coeffs=coef.glmnet(cvfit,'lambda.min')

#build a new model on whole training set & test on test set
#cvfit_new <- fit_enet(data, design, target, nparamfolds, nlam, lammin)
cvfit_new <- fit_enet(data, design, target, nparamfolds, lamgrid)
Test_Rsq_new <- fit_test(cvfit_new, tdesign, ttarget, k)

#create holders for results
maxRsq_list <- list()
meanRsq_list <- list()
Test_Rsq_list <- list()
Test_Rsq_new_list <- list()
Best_model_list <- list()
  
#fill holders with results
maxRsq_list <- maxRsq
meanRsq_list <- meanRsq
Test_Rsq_list <- Test_Rsq
Test_Rsq_new_list <- Test_Rsq_new
Best_model_list <- cvfit

output_list <-list(maxRsq_list, meanRsq_list, Test_Rsq_list, Test_Rsq_new_list, Best_model_list)
return(output_list)
}

