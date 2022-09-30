# -----------------------------
#           Exercise 1
# -----------------------------

# First part
exercise1a <- function(y){
  n = length(y)
  if (all((y - as.integer(y))==0) == FALSE | 0 %in% y)
  {
    print("Wrong input, all values within the given vector must be non-zero integers.")
    return()
  }
  else
  {
    firstdig <- rep(0,n) #initialization
    
    for (i in 1:n)
    {
      component <- abs(y[i])
      n_dig <- floor(log10(component)) + 1
      firstdig[i] <- floor(component/(10^(n_dig-1)))
    }
    
  }
  return(firstdig)
}

# Second part
exercise1b <- function(y){
  n = length(y)
  if (all((y - as.integer(y))==0) == FALSE | 0 %in% y)
  {
    print("Wrong input, all values within the given vector must be non-zero integers.")
    return()
  }
  else
  {
    firstdig <- as.integer(substr(abs(y),1,1))
  }
  return(firstdig)
}


# -----------------------------
#           Exercise 2
# -----------------------------

# First part
exercise2a <- function(X,Y){
  n <- length(X)
  m <- length(Y)
  if (n != m)
  {
    print("There is an inconsistency in the input vectors' dimensions.")
    return()
  }
  else
  {
    return((1/n)*sum(abs(sort(Y)-sort(X))))
  }
}

# Second part
exercise2b <- function(){
  bi <- pexp(0.3, 5, lower.tail = TRUE)
  bii <- qexp(0.1, 5, lower.tail = TRUE)
  biii <- dexp(0.09, 5)
  return(c(bi,bii,biii))
}

exercise2b()

# -----------------------------
#           Exercise 3
# -----------------------------

# First part

paignio <- function(x){
  if (x <= 0)
  {
    print("You can't bet zero or less euros.")
    return()
  }
  else
  {
    dice_rolls <- 0
    repeat
    {
      first_roll <- sample(1:6,1)
      dice_rolls <- dice_rolls + 1
      if (first_roll == 5)
      {
        return(c(-x,dice_rolls))
      }
      second_roll <- sample(1:6,1)
      dice_rolls <- dice_rolls + 1
      if (second_roll == 5)
      {
        return(c(-x,dice_rolls))
      } else if (first_roll == 3 & second_roll == 6)
      {
        return(c(x,dice_rolls))
      }
    }
  }
}

# Second part

exercise3b <- function(N){
  if (floor(N) != N | N <= 0)
  {
    print("You can't play the game zero or less times.")
    return()
  }
  else
  {
    counter <- 0
    for (i in 1:N)
    {
      result <- paignio(1)
      if (result[2] > 2)
      {
        counter <- counter + 1
      }
    }
    return(counter/N)
    # or return(100*counter/N) for a % probability
  }
}