N <- 5000 #sample size of each dataset (define here to give you potential to change across experiments)
sigma <- 0.5 #you will change this across experiments to take values of 0, 0.25, 0.25 (i.e. you repeat the entire simulation, for each sigma in-turn)

data1 <- data.frame(int = 1, #intercept
                    x1=rnorm(N),
                    x2=rnorm(N),
                    x3=rnorm(N),
                    x4=rnorm(N),
                    x5=rnorm(N))
data2 <- data.frame(int = 1, #intercept
                    x1=rnorm(N),
                    x2=rnorm(N),
                    x3=rnorm(N),
                    x4=rnorm(N),
                    x5=rnorm(N))
data3 <- data.frame(int = 1, #intercept
                    x1=rnorm(N),
                    x2=rnorm(N),
                    x3=rnorm(N),
                    x4=rnorm(N),
                    x5=rnorm(N))
data4 <- data.frame(int = 1, #intercept
                    x1=rnorm(N),
                    x2=rnorm(N),
                    x3=rnorm(N),
                    x4=rnorm(N),
                    x5=rnorm(N))
data5 <- data.frame(int = 1, #intercept
                    x1=rnorm(N),
                    x2=rnorm(N),
                    x3=rnorm(N),
                    x4=rnorm(N),
                    x5=rnorm(N))
data6 <- data.frame(int = 1, #intercept
                    x1=rnorm(N),
                    x2=rnorm(N),
                    x3=rnorm(N),
                    x4=rnorm(N),
                    x5=rnorm(N))

#Define the predictor-outcome associations across each population, where
#the N(0, sigma) is across populations, so we generate 6 (one per population/dataset). In the 
#below matrix, each row corresponds to the 6 coefficients for each population:
Coef_Matrix <- cbind( -2, #this ensures that there isnt a 50% event prevalence - its around 16% with \beta_0 being -2
                      log(2) + rnorm(6, sd = sigma), 
                      log(1.5) + rnorm(6, sd = sigma), 
                      log(1.5) + rnorm(6, sd = sigma), 
                      log(1.5) + rnorm(6, sd = sigma), 
                      log(1.5) + rnorm(6, sd = sigma)) #when sigma = 0, all populations have same covaraite-outcome association

#Generate y outcomes depends on sigma:
step_lp <- function(df, Coefs){
  lp <- as.numeric(data.matrix(df) %*% Coefs)
  print(lp)
  pi <- exp(lp)/(1+exp(lp))
  y <- rbinom(n = nrow(df), size = 1, prob = pi)
  
  df$outcomes <- y
  return(df)
}

data1 <- step_lp(df = data1, Coefs = Coef_Matrix[1,])
data2 <- step_lp(df = data2, Coefs = Coef_Matrix[2,])
data3 <- step_lp(df = data3, Coefs = Coef_Matrix[3,])
data4 <- step_lp(df = data4, Coefs = Coef_Matrix[4,])
data5 <- step_lp(df = data5, Coefs = Coef_Matrix[5,])
data6 <- step_lp(df = data6, Coefs = Coef_Matrix[6,])
setwd("D:/sigma_0.51")
write.csv(data1,"data1.csv")
write.csv(data2,"data2.csv")
write.csv(data3,"data3.csv")
write.csv(data4,"data4.csv")
write.csv(data5,"data5.csv")
write.csv(data6,"data6.csv")

library(philentropy)


head(data1)
head(data2)
head(data3)
head(data4)



