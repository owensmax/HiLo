fit_test=function(cvfit, valdesign, valtarget, k ){

#set number of folds
k <- 5

#set params
mod <- 'lambda.min'
nt <- length(valtarget)

#make predictions
pred<- predict(cvfit,valdesign,s=mod)

#assess accuracy of predictions
SStotal <- sum((valtarget - mean(valtarget))^2)
SStotal2 <- var(valtarget)
SSresid <- sum((valtarget - pred)^2)
Rsq = 1 - SSresid/SStotal

#store coefficients from final model

#calculate degrees of freedom and AIC
coeffs=coef.glmnet(cvfit,mod)
DF = nnzero(coeffs)+1
AIC = nt*log(SSresid/nt,DF) + 2*k
AICc = AIC + 2*k*(k+1)/(nt-k-1)

return(Rsq)
}
