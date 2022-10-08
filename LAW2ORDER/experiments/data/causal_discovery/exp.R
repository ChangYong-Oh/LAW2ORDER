library(graph)
library(bnlearn)
source("topological_ordering.R")
source("causal_discovery_pcalg.R")
source("normalized_marginal_likelihood.R")
source("generate_data.R")
source("pc_given_permutation.R")
set_data("networks/sachs.dsc")
set_data("networks/child.dsc")
set_data("networks/insurance.dsc")
dm.bn = sachs.train
dm.n = dim(dm.bn)[1]
n_variable <- dim(dm.bn)[2]
alpha = 0.01

permutation <- sample(n_variable, n_variable)
pc_given_permutation(dm.bn = dm.bn, permutation = permutation)

run_time_check(mildew.train, 1)

permutation_score(permutation = permutation, dm.bn = dm.bn, alpha = alpha)


g.bn <- pc.stable(dm.bn, alpha = alpha)


dm.pcalg <- data_bnlearn2pcalg(dm.bn)
suffStat <- list(dm = dm.pcalg$dm, nlev = as.integer(dm.pcalg$nlev), adaptDF = FALSE)
V <- colnames(dm.pcalg$dm)
start_time <- Sys.time()
g.pcalg <- as(pc(suffStat = suffStat, indepTest = disCItest, alpha = alpha, labels = V)@graph, "graphAM")
print(Sys.time() - start_time)
print(g.pcalg)


pval_mat <- marginal_independence_test(dm.bn)
g <- determine_edge(permutation = permutation, dm.bn = dm.bn, marginal_independence_p_values = pval_mat, alpha = alpha)

g.pc <- run_pcalg.pc(dm.bn = dm.bn, alpha = alpha)

decay <- 0.01
topological_order_score_regression(permutation = permutation, dm = dm.bn, decay = decay, valid.n = round(dm.n / 10), seed = 0)

cnt <- 1
for (cn in colnames(dm.bn)){
  n_cat <- dim(table(dm.bn[cn]))
  print(sprintf('%20s %2d', cn, n_cat))
  cnt <- cnt * n_cat
}
print(cnt)
print(log(cnt))
print(dm.n + cnt)
print(dm.n * log(dm.n) * log(cnt))



library(bnlearn)
source("generate_data.R")
source("pc_given_permutation.R")
set_data("networks/sachs.dsc")
dm.bn = sachs.train
dm.n = dim(dm.bn)[1]
n_variable <- dim(dm.bn)[2]
permutation <- sample(n_variable, n_variable)
adj_mat <- pc_given_permutation(dm.bn, permutation)


library(bnlearn)
dataname <- 'sachs'
original_bn <- bnlearn::read.dsc(sprintf('networks/%s.dsc', dataname))
original_adjmat <- as.graphAM(original_bn)@adjMat