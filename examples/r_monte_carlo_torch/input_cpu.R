set.seed(42)

samples <- 50000
x <- runif(samples)
y <- runif(samples)
inside <- x * x + y * y <= 1
estimate <- 4 * mean(inside)

print(estimate)
