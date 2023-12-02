#our data mining objective is to find patterns from the 12,330 sessions in the dataset to 
#predict the intention of the customers to purchase the product and generate revenue 
#based on the actions they have performed in the website.
#Along with intention of the customer to generate revenue we also tend to identify the 
#target group and develop business model accordingly.

str(df)
#describing the data
#The dataset consists of 10 numerical and 8 categorical attributes
#Bounce rate: The value of "Bounce Rate" feature for a web page refers to the percentage of visitors who enter the site from that page and then leave ("bounce") without triggering any other requests to the analytics server during that session
#Exit rate:The value of "Exit Rate" feature for a specific web page is calculated as for all pageviews to the page, the percentage that were the last in the session.
#Page Value:The "Page Value" feature represents the average value for a web page that a user visited before completing an e-commerce transaction
#Special Day:The "Special Day" feature indicates the closeness of the site visiting time to a specific special day in which the sessions are more likely to be finalized with transaction
rm(list=ls())



install.packages("moments")
install.packages("corrplot")
library(moments)
library(corrplot)
library(rpart)
library(rpart.plot)

# Set working directory ####
setwd("~/Downloads/Documents/UTD/BUAN 6356/Project")

df <- read.csv("Online_Shopping.csv")

View(df)

summary(df)
names(df[,c(11,16,17,18)])
df.new <- df[,-c(11,16,17,18)]
View(df.new)

boxplot(df.new)

#number of different types of pages visited by the visitor

#are the duration in minutes or seconds?

hist(df$Administrative,breaks = 30)
table(df$Administrative)
table(df$Informational)
table(df$ProductRelated)

table(df$SpecialDay,df$Month)
#may and feb are potential months that sessions are more likely to be finalized with transaction

table(df$VisitorType)
#number of visitors

table(df$VisitorType,df$Weekend) 
#probability of new visitor visiting on weekends = 0.28
#probability of old visitor visiting on weekends = 0.22

table(df$VisitorType,df$Revenue) 
#probability of new visitor purchasing = 0.24
#probability of old visitor purchasing = 0.13

table(df$VisitorType,df$Revenue,df$Weekend) #make statements here if necessary

plot <- cor(df.new)
corrplot(plot)
#exit rates and bounce rates are strongly co-related

#########################(descision tree) with 1:1

install.packages("dplyr")
library(dplyr)

false.data <- read.csv("false_records.csv")
true.data <- read.csv("true_records.csv")

reduced.df <- rbind(false.data,true.data)
View(reduced.df)
summary(reduced.df)
table(reduced.df$revenue)
names(reduced.df[,c(7,8,9,11:15)])
reduced.df.new <- reduced.df[,-c(7,8,9,11:15)]
#remove 15 column
View(reduced.df.new)

set.seed(1) #to get same dataset
dim(reduced.df.new)
train.index <- sample(c(1:3800), 3800*0.5)
train.df <- reduced.df.new[train.index, ]
valid.df <- reduced.df.new[-train.index, ]
head(train.df, 5)
View(train.df)

#building the model
build.dt <- rpart(revenue ~ .,data=train.df,method = "class")
rpart.plot(build.dt)
print(build.dt)
predict(build.dt, type='class')
prp(build.dt, type=1,varlen = 30)
?prp

library(caret)
library(e1071)

#applying training data to the built model and creating confusion matrix for accuracy
train.build <- predict(build.dt,train.df,type="class")
confusionMatrix(train.build,as.factor(train.df$revenue))

#applying validation data to the built model and creating confusion matrix for accuracy
valid.build <- predict(build.dt,valid.df,type="class")
confusionMatrix(valid.build,as.factor(valid.df$revenue))



#finding ROC andn area under curve = 0.778
library(pROC)
dt.roc <- predict(build.dt,valid.df,type = "prob")
roc.dt <- roc(valid.df$revenue,dt.roc[,1])
plot.roc(roc.dt)
auc(roc.dt)

######################applying entire old data to the built model and creating confusion matrix for accuracy
data_re_read <- read.csv("Online_Shopping.csv")
names(data_re_read[,c(7,8,9,11:15)])
entire.df <- data_re_read[,-c(7,8,9,11:15)]
colnames(entire.df) <- c("administrative", "administrative_duration","informational","informational_duration","product_related","product_related_duration","special_day","visitor_type","weekend","revenue")
View(entire.df)

entire.build <- predict(build.dt,entire.df,type="class")
confusionMatrix(entire.build,as.factor(entire.df$revenue))

#finding ROC andn area under curve = 0.6763
entire.dt.roc <- predict(build.dt,entire.df,type = "prob")
entire.roc.dt <- roc(entire.df$revenue,entire.dt.roc[,1])
plot.roc(entire.roc.dt)
auc(entire.roc.dt)


#############################################################(clustering)
cluster.data <- read.csv("Online_Shopping.csv")
View(cluster.data)
names(cluster.data[,c(7:9,11:18)])
reduced.cluster.data <- cluster.data[,-c(7:9,11:18)]
View(reduced.cluster.data)
normalize.df <- sapply(reduced.cluster.data, scale)
View(normalize.df)

set.seed(1)
km.df <- kmeans(normalize.df,4 )
km.df$cluster
km.df$centers

plot(c(0), xaxt = 'n', ylab = "", type = "l", 
     ylim = c(min(km.df$centers), max(km.df$centers)), xlim = c(0, 7))

axis(1, at = c(1:7), labels = names(reduced.cluster.data))

for (i in c(1:7))
  lines(km.df$centers[i,], lty = i, lwd = 2, col = ifelse(i %in% c(1,3, 5),"blue", "red"))

text(x = 0.2, y = km.df$centers[, 1], labels = paste("Cluster", c(1:4)))




###############************ignore from here***************#################

#########################(logit)

data.frame(miss.val=sapply(reduced.df.new, function(x) + sum(length(which(is.na(x)))))) 
reduced.df.new.logit <- reduced.df.new
reduced.df.new.logit$rev_num <- as.integer(reduced.df.new$revenue)
names(reduced.df.new.logit)
reduced.df.new.logit <- reduced.df.new.logit[,-10]
names(reduced.df.new.logit)[10] <- "revenue"
head(reduced.df.new.logit,5)
set.seed(1)
train.index <- sample(c(1:3800), 3800*0.5)
train.df <- reduced.df.new.logit[train.index, ]
valid.df <- reduced.df.new.logit[-train.index, ]

View(train.df)
#build logit model
build.logit <- glm(revenue ~ .,data = train.df,family ="binomial" )
options(scipen=999)
summary(build.logit)

#applying training data to the built model and creating confusion matrix for accuracy
logit.training <- predict(build.logit,train.df, type = "response")
confusionMatrix(as.factor(ifelse(logit.training > 0.7, 1, 0)), as.factor(train.df$revenue))

length(logit.training2)
logit.training2 <- predict(build.logit,reduced.df.new.logit, type = "response")
confusionMatrix(as.factor(ifelse(logit.training2 > 0.7, 1, 0)), as.factor(train.df$revenue))


class(logit.training)
class(train.df)


#############################################################(to be questioned to professor)
?prcomp
pc <- prcomp(df.new,scale. = T)
summary(pc)

df2 <- read.csv("df.csv")
df3 <- df2[,-c(7,8,9,13,14,15)]
View(df3)
#after conducting pca lets consider first 10 variables to obtain the information

#pc.df <- df[,-c(7,8,9,12,13,14,15)]
#View(pc.df)
#new dataframe after pca
#class(df$Revenue)


##########################################

df4 <- df.new3[,-c(12,13,14)]
View(df4)

df5 <- df.new3[,c(12,13,14)]
View(df5)

pc <- prcomp(df4,scale. = T)
summary(pc)

df6 <- df4[,-c(9,10,11)]
View(df6)


df7 <- cbind(df6,df5)
View(df7)


set.seed(1)
train.index <- sample(c(1:3800), 3800*0.5)
train.df <- df7[train.index, ]
valid.df <- df7[-train.index, ]
View(train.df)

build.dt <- rpart(Revenue ~ .,data=train.df,method = "class")
prp(build.dt)

table(df7$Administrative)



#prblem occurs when we normalize the data
