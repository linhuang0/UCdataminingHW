library(rpart)       # For fitting regression tree
library(tree) 
library(randomForest) # For random forest
library(gbm)         # For gradient boosting
library(arules)
library(MASS)
library(ggplot2)
library(cluster)
library(kernlab)
library(e1071)
#1.(10 marks) In this question, you will fit regression trees to predict sales using the Carseats data. 
#This dataset has been divided into training and testing sets: carseatsTrain.csv and carseatsTest.csv (download these sets from Learn). 
#Use the tree(), randomForest() and gbm() R functions to answer this question (see Section 8.3 of the course textbook).
### (a) 
# Read the training and testing data
carseatsTrain <- read.csv("carseatsTrain.csv")
carseatsTest <- read.csv("carseatsTest.csv")

carseatsTrain$Urban <- ifelse(carseatsTrain$Urban == "No", 0, 1)
carseatsTrain$US <- ifelse(carseatsTrain$US == "No", 0, 1)

carseatsTest$Urban <- ifelse(carseatsTest$Urban == "No", 0, 1)
carseatsTest$US <- ifelse(carseatsTest$US == "No", 0, 1)

#Fit a regression tree
tree.carseats <- tree(Sales ~ ., data = carseatsTrain)

#Interpret the results: The unpruned tree that results from top-down greedy splitting on the training data is shown. This resulting tree might be too complex.
summary(tree.carseats)

#Notice that the output of summary() indicates that only 6 of the variables have been used in constructing the tree. In the context of a regression tree, the deviance is simply the sum of squared errors for the tree. We now plot the tree.
plot(tree.carseats)
text(tree.carseats, pretty = 0)
# Predict on training set
train.pred <- predict(tree.carseats, newdata = carseatsTrain)
train.mse <- mean((train.pred - carseatsTrain$Sales)^2)
cat("Unpruned Tree Training MSE is:", train.mse, "\n")

# Predict on test set
test.pred <- predict(tree.carseats, newdata = carseatsTest)
test.mse <- mean((test.pred - carseatsTest$Sales)^2)
cat("Unpruned Tree Test MSE is:", test.mse, "\n")
### (b) 
# Prune the tree
set.seed(1)
cv.carseats <- cv.tree(tree.carseats)
names(cv.carseats)
par(mfrow = c(1, 1))
plot(cv.carseats$size, cv.carseats$dev, type = "b")
plot(cv.carseats$k, cv.carseats$dev, type = "b")

par(mfrow = c(1,1))
#It seems that the 17th is the best
prune.car <- prune.tree(tree.carseats, best = 17)
plot(prune.car)
text(prune.car, pretty = 0)

#Estimate the error of the tree
predict.prune.train <- predict(prune.car, newdata = carseatsTrain)
prune.mse.train<-mean((predict.prune.train - carseatsTrain$Sales)^2)
cat("Pruned Tree Training MSE is:", prune.mse.train, "\n")

predict.prune.test <- predict(prune.car, newdata = carseatsTest)
testprune.mse.test<-mean((predict.prune.test - carseatsTest$Sales)^2)
cat("Pruned Tree Test MSE is:", testprune.mse.test, "\n")

### (c) 
set.seed(1)
bag.carseats <- randomForest(Sales ~ ., data = carseatsTrain, mtry = 9,importance = TRUE)
bag.carseats

#How well does this bagged model perform on the test set
train.predict.bag <- predict(bag.carseats, newdata = carseatsTrain)
train.bag.mse<-mean((train.predict.bag - carseatsTrain$Sales)^2)
cat("Training MSE for bagged is:", train.bag.mse, "\n")

test.predict.bag <- predict(bag.carseats, newdata = carseatsTest)
test.bag.mse<-mean((test.predict.bag - carseatsTest$Sales)^2)
cat("Test MSE for bagged is:", test.bag.mse, "\n")

set.seed(1)
bag.carseats.mtry3 <- randomForest(Sales ~ ., data = carseatsTrain, mtry = 3,importance = TRUE)

train.predict.bag.mtry3 <- predict(bag.carseats.mtry3, newdata = carseatsTrain)
train.bag.mse.mtry3<-mean((train.predict.bag.mtry3 - carseatsTrain$Sales)^2)
cat("RandomForest mtry = 3 Training MSE is:", train.bag.mse.mtry3, "\n")

test.predict.bag.mtry3 <- predict(bag.carseats.mtry3, newdata = carseatsTest)
test.bag.mse.mtry3<-mean((test.predict.bag.mtry3 - carseatsTest$Sales)^2)
cat("RandomForest  mtry = 3 Test MSE is:", test.bag.mse.mtry3, "\n")

#Use mtry = 5
set.seed(1)
bag.carseats.mtry5 <- randomForest(Sales ~ ., data = carseatsTrain, mtry = 5,importance = TRUE)

train.predict.bag.mtry5 <- predict(bag.carseats.mtry5, newdata = carseatsTrain)
train.bag.mse.mtry5<-mean((train.predict.bag.mtry5 - carseatsTrain$Sales)^2)
cat("RandomForest mtry = 5 Training MSE is:", train.bag.mse.mtry5, "\n")

test.predict.bag.mtry5 <- predict(bag.carseats.mtry5, newdata = carseatsTest)
test.bag.mse.mtry5<-mean((test.predict.bag.mtry5 - carseatsTest$Sales)^2)
cat("RandomForest  mtry = 5 Test MSE is:", test.bag.mse.mtry5, "\n")


#d
#In boosting, unlike in bagging, the construction of each tree depends strongly on the trees that have already been grown
set.seed(1)
boost.carseat<-gbm(Sales~.,data=carseatsTrain,distribution="gaussian",n.trees=5000,interaction.depth=8)
summary(boost.carseat)

### (d)
#In boosting, unlike in bagging, the construction of each tree depends strongly on the trees that have already been grown
set.seed(1)
boost.carseat<-gbm(Sales~.,data=carseatsTrain,distribution="gaussian",n.trees=5000,interaction.depth=8)
summary(boost.carseat)
#Use the boosted model to predict sales on the test set:
yhat.carseat.train<-predict(boost.carseat,newdata=carseatsTrain,n.trees=5000)
boosted.mse.train<-mean((yhat.carseat.train-carseatsTrain$Sales)^2)
cat("Boosted model 5000 Training MSE is:", boosted.mse.train, "\n")

yhat.carseat.test<-predict(boost.carseat,newdata=carseatsTest,n.trees=5000)
boosted.mse.test<-mean((yhat.carseat.test-carseatsTest$Sales)^2)
cat("Boosted model 5000 Test MSE is:", boosted.mse.test, "\n")
set.seed(1)
boost.carseat<-gbm(Sales~.,data=carseatsTrain,distribution="gaussian",n.trees=2000,interaction.depth=8)
summary(boost.carseat)

#Use the boosted model to predict sales on the test and training set:
yhat.carseat.train<-predict(boost.carseat,newdata=carseatsTrain,n.trees=2000)
boosted.mse.train<-mean((yhat.carseat.train-carseatsTrain$Sales)^2)
cat("Boosted model 2000 Training MSE is:", boosted.mse.train, "\n")

yhat.carseat.test<-predict(boost.carseat,newdata=carseatsTest,n.trees=2000)
boosted.mse.test<-mean((yhat.carseat.test-carseatsTest$Sales)^2)
cat("Boosted model 2000 Test MSE is:", boosted.mse.test, "\n")

