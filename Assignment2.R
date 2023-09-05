# Load the training data
train_data <- read.csv("BankTrain.csv")
# Fit the logistic regression model
model <- glm(y ~ x1 + x2, data = train_data, family = binomial()) 
# Summarize the model
summary(model)
