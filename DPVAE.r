library('Matrix')
library('MASS')
source("https://raw.githubusercontent.com/ggrothendieck/gsubfn/master/R/list.R")

neuralnet.load <- function(path){
  fc1_W  <<- matrix(scan(paste(path,'fc1_W',sep='/')), 400, 784, byrow=TRUE)
  fc1_b  <<- scan(paste(path,'fc1_b',sep='/'))
  fc21_W <<- matrix(scan(paste(path,'fc21_W',sep='/')), 15, 400, byrow=TRUE)
  fc21_b <<- scan(paste(path,'fc21_b',sep='/'))
  fc22_W <<- matrix(scan(paste(path,'fc22_W',sep='/')), 15, 400, byrow=TRUE)
  fc22_b <<- scan(paste(path,'fc22_b',sep='/'))
  fc3_W  <<- matrix(scan(paste(path,'fc3_W',sep='/')), 400, 15, byrow=TRUE)
  fc3_b  <<- scan(paste(path,'fc3_b',sep='/'))
  fc4_W  <<- matrix(scan(paste(path,'fc4_W',sep='/')), 784, 400, byrow=TRUE)
  fc4_b  <<- scan(paste(path,'fc4_b',sep='/'))
}

neuralnet.sigmoid <- function(x){
  1/(1+exp(-x))
}

neuralnet.relu  <- function(x){
  structure(vapply(x, function(z) max(0,z), numeric(1)),dim=dim(x))
}

neuralnet.dense <- function(x, W, b, act=""){
  if (act=='sigmoid')
    neuralnet.sigmoid(x%*%t(W)+b)
  else if (act=='relu')
    neuralnet.relu(x%*%t(W)+b)
  else
     x%*%t(W)+b
}

encoder.encode <- function(x){
  h1 <- neuralnet.dense(x, fc1_W, fc1_b, 'relu')
  list(neuralnet.dense(h1, fc21_W, fc21_b), neuralnet.dense(h1, fc22_W, fc22_b))
}

decoder.decode <- function(x){
  h3 <- neuralnet.dense(x, fc3_W, fc3_b, 'relu')
  neuralnet.dense(h3, fc4_W, fc4_b, 'sigmoid')
}

reparametrize <- function(mean_, logvar){
  n <- dim(mean_)[1]
  sapply(1:n, function(i){mvrnorm(mu=mean_[i,], Sigma=Diagonal(x = exp(logvar[i,])))})
}

VDP.load <- function(path){
  weights_ <<- scan(paste(path, 'weights', sep='/'))
  mean_    <<- matrix(scan(paste(path, 'means', sep='/')), 500, 15, byrow=TRUE)
  var_     <<- matrix(scan(paste(path, 'covars', sep='/')), 500, 15, byrow=TRUE)
}

VDP.sample <- function(n){
  z <- max.col(t(rmultinom(n, 1, weights_)))
  sapply(z, function(i){mvrnorm(mu=mean_[i,], Sigma=Diagonal(x=var_[i,]))})
}

init <- function(path){
  VDP.load(path)
  neuralnet.load(path)
}

sample <- function(n, display=FALSE){
  ret <- decoder.decode(t(VDP.sample(n)))
}

reconstruct <- function(x, display=FALSE){
  list[mean_, logvar] <- encoder.encode(x)
  z   <- t(reparametrize(mean_, logvar))
  ret <- decoder.decode(z)
}

display <- function(x){
    x <- structure(x, dim=c(dim(x)[1], 28, 28))
    x[,,28:1]
}
