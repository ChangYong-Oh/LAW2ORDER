library(bnlearn)
library(pcalg)

disCItest_sprint_error <- function(c) {
  if (c$message == "invalid format '%d'; use format %f, %e, %g or %a for numeric objects"){
    return(1)
  } else {
    stop(c$message)
  }
}


marginal_independence_test <- function(dm.bn){
  dm.pcalg <- data_bnlearn2pcalg(dm.bn)
  suffStat <- list(dm = dm.pcalg$dm, nlev = as.integer(dm.pcalg$nlev), adaptDF = FALSE)
  node.names <- colnames(dm.pcalg$dm)
  n_variables <- dim(dm.pcalg$dm)[2]
  p_value_mat <- matrix(0, nrow = n_variables, ncol = n_variables)
  for (i in 1:(n_variables-1)){
    for (j in (i+1):n_variables){
      ci <- pcalg::disCItest(x = i, y = j, S = NULL, suffStat = suffStat)
      p_value_mat[i, j] <- p_value_mat[j, i] <- ci
    }
  }
  rownames(p_value_mat) <- colnames(p_value_mat) <- 1:n_variables
  return(p_value_mat)
}


determine_edge <- function(permutation, dm.bn, marginal_independence_p_values, alpha) {
  dm.pcalg <- data_bnlearn2pcalg(dm.bn)
  suffStat <- list(dm = dm.pcalg$dm, nlev = as.integer(dm.pcalg$nlev), adaptDF = FALSE)
  node.names <- colnames(dm.pcalg$dm)
  n_variables <- dim(dm.pcalg$dm)[2]
  adjmat <- matrix(0, nrow = n_variables, ncol = n_variables)
  src <- as.integer(permutation[1])
  dst <- as.integer(permutation[2])
  ci <- marginal_independence_p_values[src, dst]
  if (ci < alpha) {
    adjmat[1, 2] <- 1
  }
  for (j in 3:n_variables) {
    dst <- as.integer(permutation[j])
    for (i in 1:(j-1)){
      src <- as.integer(permutation[i])
      ci0 <- marginal_independence_p_values[src, dst]
      if (ci0 < alpha) {
        src_prt <- permutation[which(adjmat[1:(i-1), i] == 1, arr.ind = 1)]
        ci1 <- tryCatch(pcalg::disCItest(x = src, y = dst, S = src_prt, suffStat = suffStat), error = disCItest_sprint_error)
        # print(sprintf('from %d to %d', i, j))
        # print(sprintf('SRC : %s', src))
        # print(sprintf('DST : %s', dst))
        # print(src_prt)
        if (ci1 < alpha) {
          src_ptl_cdr <- permutation[which(adjmat[i, (i+1):(j-1)] == 1, arr.ind = 1) + i]
          ci2 <- tryCatch(pcalg::disCItest(x = src, y = dst, S = c(src_prt, src_ptl_cdr), suffStat = suffStat), error = disCItest_sprint_error)
          # print(c(src_prt, src_ptl_cdr))
          if (ci2 < alpha) {
            src_ptl_cdr_prt <- c()
            for (src_chd in src_ptl_cdr){
              src_ptl_cdr_prt <- c(src_ptl_cdr_prt, permutation[which(adjmat[1:(src_chd-1), src_chd] == 1, arr.ind = 1)])
            }
            conditioning <- unique(c(src_prt, src_ptl_cdr, src_ptl_cdr_prt))
            conditioning <- as.integer(conditioning[conditioning != src])
            ci3 <- tryCatch(pcalg::disCItest(x = src, y = dst, S = conditioning, suffStat = suffStat), error = disCItest_sprint_error)
            # print(conditioning)
            if (ci3 < alpha) {
              adjmat[i, j] <- 1
            }
          }
        }
      } 
      # print('ci.test has been done')
    }
  }
  # print("CI tests completed")
  adjmat[permutation, permutation] <- adjmat
  rownames(adjmat) <- colnames(adjmat) <- node.names
  graph <- graph::graphAM(adjMat = adjmat, edgemode = 'directed', values = list(weight = 1))
  return(graph)
}


data_split <- function(dm, valid.n, seed = 0){
  dm.n = dim(dm)[1]
  train.n = dm.n - valid.n
  set.seed(seed)
  shuffled_ind = sample(dm.n, dm.n, replace = FALSE)
  set.seed(NULL)
  train.ind = shuffled_ind[1:train.n]
  valid.ind = shuffled_ind[(train.n+1):dm.n]
  train.dm = dm[train.ind, ]
  valid.dm = dm[valid.ind, ]
  return(list(train = train.dm, valid = valid.dm))
}


cv_loglik <- function(dm, bn, cv.n = 10, seed = 0){
  set.seed(seed)
  seed_list = sample(10000, size = cv.n, replace = FALSE)
  set.seed(NULL)
  dm.n = dim(dm)[1]
  valid.n = round(dm.n / cv.n)
  cv_scores = c()
  for (s in 1:cv.n) {
    train_valid = data_split(dm, valid.n = valid.n, seed = seed_list[s])
    train.dm = train_valid$train
    valid.dm = train_valid$valid
    fit <- bnlearn::bn.fit(bn, data = train.dm)
    cv_scores <- c(cv_scores, logLik(fit, data = valid.dm) / valid.n)
    print(sprintf('%s %2d has been done out of %d', Sys.time(), s, cv.n))
  }
  return(mean(cv_scores))
}


topological_order_score <- function(permutation, dm.bn, marginal_independence_p_values, alpha) {
  start_time <- Sys.time()
  g <- determine_edge(permutation = permutation, dm.bn = dm.bn, marginal_independence_p_values = marginal_independence_p_values, alpha = alpha)
  print(g)
  print("determine_edge")
  print(Sys.time() - start_time)
  start_time <- Sys.time()
  s <- cv_loglik(dm = dm.bn, bn = bnlearn::as.bn(g))
  print("cv_loglik")
  print(Sys.time() - start_time)
  return(s)
}


regress_on_parents <- function(permutation, i, dm, decay){
  response <- dm[,permutation[i]]
  predictors <- as.data.frame(dm[,permutation[1:(i-1)]])
  colnames(predictors) <- colnames(dm)[permutation[1:(i-1)]]
  if (sum(table(response) > 0) > 1) {
    multinom_model <- nnet::multinom(response ~ ., data = predictors, decay = decay, trace = FALSE)
    return(multinom_model)
  }
  return(NULL)
}

evaluate_regression <- function(model, permutation, i, dm){
  response <- dm[,permutation[i]]
  predictors <- as.data.frame(dm[,permutation[1:(i-1)]])
  colnames(predictors) <- colnames(dm)[permutation[1:(i-1)]]
  pred_class <- predict(model, newdata = predictors, "class")
  pred_prob <- predict(model, newdata = predictors, "probs")
  if (is.null(dim(pred_prob))){
    case.large <- pred_class[pred_prob > 0.5][1]
    case.small <- pred_class[pred_prob <= 0.5][1]
    if (is.na(case.large)) {
      case.small <- toString(case.small)
      case.large <- toString(levels(pred_class)[levels(pred_class) != case.small])
    } 
    if (is.na(case.small)) {
      case.large <- toString(case.large)
      case.small <- toString(levels(pred_class)[levels(pred_class) != case.large])
    }
    case.small <- toString(case.small)
    case.large <- toString(case.large)
    
    if (pred_prob[1] > 0.5){
      pred_prob_colname <- c(case.large, case.small)
    } else {
      pred_prob_colname <- c(case.small, case.large)
    }
    pred_prob <- cbind(pred_prob, 1 - pred_prob)
    colnames(pred_prob) <- pred_prob_colname
    # print('Except 2 class case')
    # print(levels(pred_class))
    # print(colnames(pred_prob))
  }
  if (length(levels(response)) != dim(pred_prob)[2]) {
    min_pred_prob <- min(pred_prob[pred_prob > 0])
    for (name in levels(response)) {
      if (!(name %in% colnames(pred_prob))) {
        aug_matrix <- matrix(min_pred_prob / 10, nrow = dim(pred_prob)[1], ncol = 1)
        colnames(aug_matrix) <- c(name)
        pred_prob <- cbind(pred_prob, aug_matrix)
      }
    }
    pred_prob <- pred_prob[, levels(response)]
  }
  if (length(levels(response)) != dim(pred_prob)[2]) {
    stop("Mismatch")
  }
  response.one_hot <- model.matrix(~0+response)
  response.one_hot.colnames <- c()
  for (i in 1:dim(pred_prob)[2]){
    response.one_hot.colnames <- c(response.one_hot.colnames, paste0('response', colnames(pred_prob)[i]))
  }
  # print(levels(response))
  # print(response.one_hot.colnames)
  # print(colnames(response.one_hot))
  response.one_hot <- response.one_hot[, response.one_hot.colnames]
  crossentropy <- mean(rowSums(response.one_hot * log(pred_prob)))
  return(crossentropy)
}


topological_order_regression_validation_score <- function(permutation, dm.train, dm.valid, decay){
  n_variables <- dim(dm.train)[2]
  valid_score <- 0
  for (i in 2:n_variables) {
    multinom_model <- regress_on_parents(permutation = permutation, i = i, dm = dm.train, decay = decay)
    if (!is.null(multinom_model)) {
      eval_score <- evaluate_regression(model = multinom_model, permutation = permutation, i = i, dm = dm.valid)
      valid_score <- valid_score + eval_score
    }
  }
  return(valid_score)
}

topological_order_score_regression <- function(permutation, dm, decay, valid.ratio, n_split, seed){
  start_time <- Sys.time()
  valid.n <- round(dim(dm)[1] * valid.ratio)
  set.seed(seed)
  seed_list = sample(10000, size = n_split, replace = FALSE)
  set.seed(NULL)
  dm.n = dim(dm)[1]
  cv_scores = c()
  print(sprintf('%s %2d has been done out of %d', Sys.time(), 0, n_split))
  for (s in 1:n_split) {
    train_valid = data_split(dm, valid.n = round(dm.n * valid.ratio), seed = seed_list[s])
    dm.train = train_valid$train
    dm.valid = train_valid$valid
    valid_score <- topological_order_regression_validation_score(
      permutation = permutation, dm.train = dm.train, dm.valid = dm.valid, decay = decay)
    cv_scores <- c(cv_scores, valid_score)
    print(sprintf('%s %2d has been done out of %d', Sys.time(), s, n_split))
  }
  print(Sys.time() - start_time)
  return(mean(cv_scores))
}

