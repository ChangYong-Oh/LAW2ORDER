suppressPackageStartupMessages({
  library(graph)
  library(bnlearn)
})

data_bnlearn2pcalg <- function(x){
  nlev <- c()
  for (i in 1:dim(x)[2]){
    x[,i] <- as.integer(as.integer(x[,i]) - 1)
    nlev <- c(nlev, length(levels(x[,i])))
  }
  V <- colnames(x)
  data_pcalg_format = list(dm = x, nlev = as.integer(nlev), V = V)
  return(data_pcalg_format)
}


N_DATA <- 10000


simulate_data <- function(network, train.n = N_DATA, test.n = N_DATA, seed = 0){
  set.seed(seed)
  df.full <- bnlearn::rbn(network, n = (train.n + test.n))
  set.seed(NULL)
  df = list(train = df.full[1:train.n, ], test = df.full[(train.n + 1):(train.n + test.n), ])
  return(df)
}


set_data <- function(network.file, envir = .GlobalEnv){
  network <- bnlearn::read.dsc(network.file)
  network.name <- tools::file_path_sans_ext(basename(network.file))
  seed <- 0
  dm <- simulate_data(network, train.n = N_DATA, test.n = N_DATA, seed = seed)
  assign(paste(network.name, "train", sep = "."), dm$train, envir = envir)
  assign(paste(network.name, "test", sep = "."), dm$test, envir = envir)
  # save_dir <- dirname(network.file)
  # save(dm, file = paste(save_dir, sprintf('%s_%d_R%04d.Rdata', network.name, N_DATA, seed), sep = "/"))
}