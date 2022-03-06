Notebook for 2019-2020 R solutions

```{r}
ask1 <- function(x,lambda){
  if (lambda <= 0 | floor(x) != x | x < 0)
  {
    print("Wrong input.")
  }
  else
  {
    px0 <- exp(1)^(-lambda)
    if (x == 0)
    {
      return(px0)
    }
    else
    {
      result <- px0
      for (i in 1:x)
      {
        result <- result*(lambda/i)
      }
    }
  }
  return(result)
}
```

```{r}
ask2 <- function(X){
  n <- length(X)
  
  ARRX <- sort(X)
  
  SUMTERM <- 0.0
  for (i in 1:n)
  {
    SUMTERM <- SUMTERM + (2*i - n - 1)*ARRX[i]
  }
  G <- (2/(n*(n-1)))*SUMTERM
  
  return(c(G,n))
}
```

```{r}

# pbeta(0.5,4,5)
# qbeta(0.8,4,5) # Because P(X>a) = 1 - P(X<=a)
# dbeta(0.2,4,5)
```

```{r}
ask3 <- function(y,B){
  # y is the vector of observations (y1,...,yN)
  # B is the number of times we repeat
  
  thetas <- rep(0,B)
  n <- length(y)
  
  for (i in 1:B)
  {
    cursam <- sample(y, n, replace = TRUE)
    
    s <- sd(cursam)
    ybar <- abs(mean(cursam))
    thetas[i] <- 100*s/ybar
  }
  
  meanthet = mean(thetas)
  mulfac = sqrt(1/(B-1))
  
  tempsum <- 0.0
  for (i in 1:B)
  {
    tempsum <- tempsum + (thetas[i]-meanthet)^2
  }
  se <- mulfac*sqrt(tempsum)
  
  
  result <- list(se,thetas)
  names(result) <- c("Standard Error","Calculated Estimators")
  return(result)
}
```
