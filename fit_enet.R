#fit_enet=function(data, design, target, nparamfolds, nlam, lammin){
fit_enet=function(data, design, target, nparamfolds, lamgrid){

#set hyperparamter alphas to test
alphas = seq(.05, 1, by=.05)

#create result holder
cvlist <- list()

# run cross-validated elastic net regression
cvfit_best <- cv.glmnet(design, target, nfolds = nparamfolds, alpha = alphas[c(1)], family = 'gaussian', lambda = lamgrid, type.measure = c("deviance"),parallel = FALSE, keep = FALSE)
cvlist[[1]] <- cvfit_best
for (i in 2:length(alphas)){
  #cvfit_new <- cv.glmnet(design, target, nfolds = nparamfolds, alpha= alphas[c(i)], 
                         #family = 'gaussian', nlambda = nlam, type.measure = c("deviance"),
                         #parallel = FALSE, keep = FALSE)
  cvfit_new <- cv.glmnet(design, target, nfolds = nparamfolds, alpha= alphas[c(i)], 
                         family = 'gaussian', lambda = lamgrid, type.measure = c("deviance"),
                         parallel = FALSE, keep = FALSE)
  cvlist[[i]] <- cvfit_new
  if (min(cvfit_new$cvm) < min(cvfit_best$cvm)){
    cvfit_best <- cvfit_new
    best_alpha <- alphas[c(i)]
    best_lambda <- cvfit_best$lambda.min}
  }

return(cvfit_best)

}
