# Data generation ----------------------------------------------

n <- 1000

x <- runif(n)
w <- 0.2
b <- 0.1

y <- w * x + b

# Model definition ---------------------------------------------

model <- function(w_hat, b_hat, x) {
  w_hat * x + b_hat
}

loss <- function(y, y_hat) {
  mean((y - y_hat)^2)
}

# Estimating via SGD --------------------------------------------

grad_loss_wrt_w <- function(x, y, y_hat) {
  mean((-2 * x) * (y - y_hat))
}

grad_loss_wrt_b <- function(x, y, y_hat) {
  mean(-2 * (y - y_hat))
}

# inicializando os pesos
w_hat <- runif(1)
b_hat <- 0

lr <- 0.001

for (step in 1:10) {
  y_hat <- model(w_hat, b_hat, x)
  w_hat <- w_hat - lr*grad_loss_wrt_w(x, y, y_hat)
  b_hat <- b_hat - lr*grad_loss_wrt_b(x, y, y_hat)
  
  if (step %% 10 == 0) {
    print(loss(y, y_hat))
  }
    
}

w_hat
b_hat



