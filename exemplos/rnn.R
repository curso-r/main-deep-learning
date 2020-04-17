# Pacotes ---------------------------------------------------------

library(keras)

# Data ------------------------------------------------------------

n <- 10000
l <- 5

cresc <- sample(c(1,0), size = n, replace = TRUE)
x <- array(dim = c(n, 5, 1))
for(i in 1:n) {
  v <- runif(2, min = -1, max = 1)
  if (cresc[i] == 1)
    x[i,,1] <- seq(from = min(v), to = max(v), length.out = 5)
  else
    x[i,,1] <- seq(from = max(v), to = min(v), length.out = 5)
}

# Model ------------------------------------------------------------

input <- layer_input(shape = c(5,1))

output <- input %>% 
  layer_simple_rnn(units = 1, input_shape = c(5,1),
                   activation = "sigmoid", use_bias = FALSE)

model <- keras_model(input, output)

model %>% compile(loss = "binary_crossentropy", 
                  optimizer = "adam",
                  metrics = "accuracy")
model %>% fit(x = x, y = cresc, epochs = 10)

# Manual calc ------------------------------------------------------

sigm <- function(x) {
  1/(1 + exp(-x))
}

w <- get_weights(model)

s <- 0
x_ <- x[1,,]
for (i in 1:5) {
  s <- sigm(x_[i]*w[[1]] + s*w[[2]])
}
s

model(x[1,,,drop=FALSE])

# Results ----------------------------------------------------------

ggplot2::qplot(predict(model, x), cresc, geom = "jitter")
model(x[1:10,,,drop=FALSE])
cresc[1:10]
