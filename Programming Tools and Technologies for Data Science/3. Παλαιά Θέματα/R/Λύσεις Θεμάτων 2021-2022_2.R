exercise1a <- function(v) {
  
  if (any(v == 0)) stop('The input contains 0s')
  if (!all(v %% 1 == 0)) stop('The input contains non-integers')
  
  # negatives <- (v < 0)
  v <- abs(v)
  logs <- log10(v)
  ndigits <- floor(logs)
  
  first_digit <- v %/% 10 ^ ndigits
  
  return(first_digit)
}


exercise1b <- function(v) {
  
  if (any(v == 0)) stop('The input contains 0s')
  if (!all(v %% 1 == 0)) stop('The input contains non-integers')
  
  return(as.integer(substr(as.character(v), 1, 1)))
}


exercise2a <- function(X, Y) {
  if (length(X) != length(Y)) stop('Non-matching dimensions')
  return(mean(abs(sort(X) - sort(Y))))
}


exercise2b <- function(rate = 5) {
  return(list(i = pexp(0.3, rate = rate),
              ii = qexp(0.1, rate = rate),
              iii = dexp(0.09, rate = rate)))
}


paignio <- function(x) {
  
  if (x <= 0) stop('x needs to be a positive number')
  
  dice <- 1:6
  prev <- 0  # 0 is like a NULL in our case
  nthrows <- 0
  
  repeat {
    cur <- sample(dice, 1)
    nthrows <- nthrows + 1
    
    if (cur == 5) {
      return(c(nthrows = nthrows, winnings = -x))
    } else if (cur == 6 & prev == 3){
      return(c(nthrows = nthrows, winnings = x))
    }
    
    prev = cur
  }
}


exercise3b <- function(N) {
  
  if (N <= 0 | N%%1 != 0) stop('N needs to be a positive integer')
  
  # Faster but more memory
  # res <- vapply(rep(1, N), paignio, numeric(2))
  # return(mean(res['nthrows',] > 2))
  
  # Slower but less memory
  nover2 <- 0
  for (i in 1:N) {
    res <- paignio(1)
    if (res['nthrows'] > 2) {
      nover2 <- nover2 + 1
    }
  }
  return(nover2 / N)
}
