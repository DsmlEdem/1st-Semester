```{r}
factt <- function(m){
  y = floor(m)
  if (y != m | m < 0)
  {
    print("I'm sorry, but this is not a natural number.")
  }
  else
  {
    vec <- rep(0.05, m)
    for (i in 1:m)
    {
      vec[i] = factorial(i)
    }
  }
  return(vec)
}
```

```{r}
# Here goes a comment
expo <- function(x,n){
  summ <- 0.0
  for (i in 1:n)
  {
    summ <- summ + (x^i)/factorial(i)
  }
  return(summ)
}
```

```{r}
alterexpo <- function(x,n){
  summ <- 0.0
  pows <- c(1:n)
  xvec <- rep(x,n)
  xvec <- xvec^pows
  return(sum(xvec/factt(n)))
}
```

```{r}
ask2 <- function(X){
  mn <- mean(X)
  n <- length(X)
  XMV <- rep(mn,n)
  return((1/n)*sum(abs(X-XMV)))
}
```

```{r}
# pnorm(2,2,1)
# qnorm(0.2,2,1)
# dnorm(0.8,2,1)
```

```{r}
perm =function(n) {
  a=NULL
  if (floor(n) != n | n <= 0)
  {
    print("Incorrect input. Not a natural number.")
  }
  else
  {
    for (i in 1:n){
    for (j in 1:n){
    for (k in 1:n){
    a=rbind(a,c(i,j,k))}}}
  }
  return(a)
}
```

```{r}
ask3b <- function(){
  MATRIX <- perm(6)
  MATel <- apply(MATRIX,1,sum)
  sumMAT <- length(MATel[MATel>11]) # number of instances
  return(sumMAT/dim(MATRIX)[1])
}
```

```{r}
ask3c <- function(m){
  counter <- 0
  for (i in 1:m)
  {
    smpls <- sample(1:6, 3, replace = TRUE)
    if (sum(smpls)>11)
    {
      counter <- counter+1
    }
    
  }
  return(counter/m)
}
```
