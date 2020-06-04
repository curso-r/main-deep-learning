# Dados ---------------------------------------

library(tidyverse)

df <- readr::read_csv(
  pins::pin("https://raw.githubusercontent.com/jbrownlee/Datasets/master/pollution.csv")
)

df <- df %>% 
  mutate(date = lubridate::make_datetime(year, month, day, hour)) %>% 
  select(-day, -month, -year, -hour, -No)

df <- df %>% 
  rename(
    pollution = pm2.5,
    dew = DEWP,
    temp = TEMP,
    press = PRES,
    wnd_dir = cbwd,
    wnd_spd = Iws,
    snow = Is,
    rain = Ir
  ) %>% 
  select(date, everything())

# limpeza - jogar fora as primeiras 24h
# substituir os NA's da poluição por 0

df <- df %>% 
  filter(row_number(date) > 24) %>% 
  mutate(pollution = ifelse(is.na(pollution), 0, pollution))

# tabelinha descritiva
df %>% 
  skimr::skim()

# grafico das séries
df %>% 
  select(-wnd_dir) %>% 
  pivot_longer(names_to = "var", values_to = "val", cols = c(-date)) %>% 
  ggplot(aes(x = date, y = val)) +
  geom_line() +
  facet_wrap(~var, scales = "free_y", ncol = 1)

df <- df %>% 
  mutate(
    cv = as.numeric(wnd_dir == "cv"),
    NE = as.numeric(wnd_dir == "NE"),
    NW = as.numeric(wnd_dir == "NW"),
    SE = as.numeric(wnd_dir == "SE")
  ) %>% 
  select(-wnd_dir)

# Preparando os dados ----------------

historico <- 365*2*24
previsao <- 7*24
pulos <- 3*24

janelas <- seq(
  from = historico, 
  to = nrow(df) - previsao -1, 
  by = pulos
)

length(janelas)
data <- df %>% arrange(date) %>% select(-date) %>% as.matrix()

x <- array(NA, dim = c(length(janelas), historico, ncol(data)))
y <- array(NA, dim = c(length(janelas), previsao, ncol(data)))

medias <- apply(data[1:historico,], 2, mean)
sds <- apply(data[1:historico,], 2, sd)

for (i in seq_along(janelas)) {
  
  janela <- janelas[i]
  
  x[i,,] <- scale(data[(janela - historico + 1):janela,], center = medias, 
                  scale = sds)
  y[i,,] <- scale(data[(janela + 1):(janela + previsao),], center = medias,
                  scale = sds)
  
}

# Definindo o modelo --------------------

library(keras)

input <- layer_input(shape = c(historico, ncol(data)))
output <- input %>% 
  layer_lstm(units = 32, return_sequences = TRUE) %>% 
  layer_average_pooling_1d(pool_size = 24) %>% 
  layer_lstm(units = 128, return_sequences = TRUE) %>% 
  layer_average_pooling_1d(pool_size = 24) %>% 
  layer_flatten() %>%
  layer_dense(units = 512, activation = "relu") %>% 
  layer_dense(units = previsao * ncol(data)) %>% 
  layer_reshape(target_shape = c(previsao, ncol(data)))
  
model <- keras_model(input, output)

model %>% compile(loss = "mse", optimizer = "sgd")

# Ajuste do modelo ---------------------------

idx <- 1:300

model %>% 
  fit(x[id,,], y[id,,], batch_size = 10, shuffle = FALSE,
      validation_data = list(x[-id,,], y[-id,,]))




  
