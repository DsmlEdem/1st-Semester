# simple linear model
n = 30 # number of samples

t.value = 2.09
2*pt(t.value, n - 2, lower.tail = FALSE) # t-test when q is positive

f.value = 943.20
pf(f.value, 1, n - 2, lower.tail = FALSE) # f-test (1, n - 2)






# multiple linear model
n = 34 # number of samples
k = 3 # number of variables

p = k + 1

## t-test
t.value = 1.11/1.054
2*pt(t.value, n - p, lower.tail = FALSE) # t-test when q is positive

## total F-test kinda useless
f.value = 39.703
pf(f.value, k, n - p, lower.tail = FALSE) # f-test (k, n - p)

# linear model prediction conf interval t-distribution
n = 32
k = 3
conf.int.level = 0.95
qt(1 - (1 - conf.int.level)/2, n - k - 1)

## nested F-test M0 < M1
n = 34
SSE0 = 31.436 # nested model with less variables
k0 = 2

SSE1 =  27.279  # model with more variables
k1 = 3

f.value = (SSE0 - SSE1) * (n - k1 -1) / SSE1 / (k1 - k0)
# if significant, choose big model
# if not significant, choose small-nested model
pf(f.value, k1 - k0, n - k1 - 1, lower.tail = FALSE) # f-test (k, n - p)






# wald
2*pnorm(0.00135/0.001430, 0, 1, lower.tail = FALSE)

# confidence interval for beta.j poisson/logistic
beta.j = -6.814e-1
se.beta.j = 1.700e-1
conf.int.level = 0.95
beta.j - qnorm(1 - (1 - conf.int.level)/2, 0, 1) * se.beta.j # left
beta.j + qnorm(1 - (1 - conf.int.level)/2, 0, 1) * se.beta.j # right

# deviance
pchisq(2.5, 1, lower.tail = FALSE) # val = deviance difference, dof = difference in num of params


