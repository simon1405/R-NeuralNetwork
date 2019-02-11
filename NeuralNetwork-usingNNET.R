##Neural Network

##Scaler

regul<- function(x){
  (x-min(x))/(max(x)-min(x))
}

normli<-function(x){
  (x-mean(x))/sd(x)
}
##fomula
library(nnet)

nnet(formula, data, weights, ...,
     subset, na.action, contrasts = NULL)

nnet(x, y, weights, size, Wts, mask,
     linout = FALSE, entropy = FALSE, softmax = FALSE,
     censored = FALSE, skip = FALSE, rang = 0.7, decay = 0,
     maxit = 100, Hess = FALSE, trace = TRUE, MaxNWts = 1000,
     abstol = 1.0e-4, reltol = 1.0e-8, ...)

mlp(x, y, size = c(5), maxit = 100,
    initFunc = "Randomize_Weights", initFuncParams = c(-0.3, 0.3),
    learnFunc = "Std_Backpropagation", learnFuncParams = c(0.2, 0),
    updateFunc = "Topological_Order", updateFuncParams = c(0),
    hiddenActFunc = "Act_Logistic", shufflePatterns = TRUE, linOut = FALSE,
    inputsTest = NULL, targetsTest = NULL, pruneFunc = NULL,
    pruneFuncParams = NULL, ...)

neuralnet(formula, data, hidden = c(1), threshold = 0.01,        
          stepmax = 1e+05, rep = 1, startweights = NULL, 
          learningrate.limit = NULL, 
          learningrate.factor = list(minus = 0.5, plus = 1.2), 
          learningrate=NULL, lifesign = "none", 
          lifesign.step = 1000, algorithm = "rprop+", 
          err.fct = "sse", act.fct = "logistic", 
          linear.output = TRUE, exclude = NULL, 
          constant.weights = NULL, likelihood = FALSE)

##split dataset
set.seed(1234)
index <- sample(c(1,2), nrow(thedata), replace = TRUE, prob = c(0.8,0.2))
train <- thedata[index == 1,]
test <- thedata[index == 2,]

err1 <- 0
err2 <- 0
for (i in 1:20){
  set.seed(1992)
  model <- nnet(diagnosis ~ ., data = train, maxit = 300, size = i, trace = FALSE)
  err1[i] <- sum(predict(model, train, type = 'class') != train$diagnosis)/nrow(train)
  err2[i] <- sum(predict(model, test, type = 'class') != test$diagnosis)/nrow(test)
}
plot(err1, type = 'b', col = 'black', lty = 2, lwd = 2, ylab = 'error', xlab = 'eps', ylim = c(0,0.05), pch = 10)
lines(err2, type = 'b', col = 'blue', lty = 2, lwd = 2, pch = 23)
legend(locator(1), legend = c('train set error rate','test set error rate'), col = c('black','blue'), lty = c(2,2), lwd = c(2,2), bty = 'n',
       pch = c(10,23))

set.seed(1992)
model_nnet <- nnet(diagnosis ~ ., data = train, maxit = 50, size = 4, trace = FALSE)
pred_nnet <- predict(model_nnet, test, type = 'class')
#accuracy
Freq_nnet <- table(test$diagnosis, pred_nnet)
Freq_nnet
accuracy_nnet <- sum(diag(Freq_nnet))/sum(Freq_nnet)
accuracy_nnet