library(bnlearn)
library(pcalg)


decimal2binary <- function(n){
  if(n > 1) {
    decimal2binary(as.integer(n/2))
  }
  cat(n %% 2)
}


pdag.undirected <- function(graph.AM){
  adj_mat <- graph.AM@adjMat
  undirected.mat <- adj_mat + t(adj_mat)
  undirected.mat[lower.tri(undirected.mat)] <- 0
  undirected.edge <- which(undirected.mat == 2, arr.ind = 2)
  undirected.edge <- undirected.edge[order(undirected.edge[,2]), ]
  undirected.edge <- undirected.edge[order(undirected.edge[,1]), ]
  undirected.n <- dim(undirected.edge)[1]
  undirected = list(edge = undirected.edge, n = undirected.n)
  return(undirected)
}


pdag.set.direction <- function(graph.AM, select.id, undirected.edge){
  undirected.n <- dim(undirected.edge)[1]
  select.vec <- rev(as.numeric(strsplit(str_pad(as.character(select.id), undirected.n, pad = "0"), "")[[1]]))
  directed.edge <- matrix(0, nrow = undirected.n, ncol = 2)
  for (r in 1:undirected.n){
    vertices = undirected.edge[r, ]
    graph.AM@adjMat[undirected.edge[r, 1], undirected.edge[r, 2]] <- 0
    graph.AM@adjMat[undirected.edge[r, 2], undirected.edge[r, 1]] <- 0
    if (select.vec[r] == 0){
      graph.AM@adjMat[min(vertices), max(vertices)] <- 1
    } else {
      graph.AM@adjMat[max(vertices), min(vertices)] <- 1
    }
  }
  return(graph.AM)
}




run_pcalg.pc <- function(dm.bn, alpha){
  dm.pcalg <- data_bnlearn2pcalg(dm.bn)
  suffStat <- list(dm = dm.pcalg$dm, nlev = as.integer(dm.pcalg$nlev), adaptDF = FALSE)
  V <- colnames(dm.pcalg$dm)
  start_time <- Sys.time()
  graph_NEL <- pcalg::pc(suffStat = suffStat, indepTest = disCItest, alpha = alpha, labels = V)@graph
  print(Sys.time() - start_time)
  return(graph_NEL)
}


run_pcalg.ges <- function(dm.bn, score){
  dm.pcalg <- data_bnlearn2pcalg(dm.bn)
  suffStat <- list(dm = dm.pcalg$dm, nlev = as.integer(dm.pcalg$nlev), adaptDF = FALSE)
  V <- colnames(dm.pcalg$dm)
  score <- new("Gauss0penObsScore", dm.pcalg$dm)
  graph <- pcalg::ges(score = score)
}


run_pcalg <- function(dm.bn, algorithm, alpha = 0.05, score = 'GaussL0penObsScore'){
  if (algorithm == "pc") {
    g <- run_pcalg.pc(dm.bn = dm.bn, alpha = alpha)
  } else if (algorithm == "ges") {
    g <- run_pcalg.ges(score = score)
  } else {
    stop("Not implemented algorithm")
  }
}



start_time <- Sys.time()

print(Sys.time() - start_time)
print(g.pclag)