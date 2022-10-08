# Title     : TODO
# Objective : TODO
# Created by: coh1
# Created on: 1/15/21

library(stringr)
library(bnlearn)
library(graph)
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


best_score_dag <- function(bn.pdag, dm){
  graph.AM <- as.graphAM(bn.pdag)
  undirected <- pdag.undirected(graph.AM = graph.AM)
  score.best <- -Inf
  select.id.best <- 0
  for (i in 0:2 ^ undirected$n - 1){
    bn.dag <- as.bn(pdag.set.direction(graph.AM = graph.AM, select.id = i, undirected.edge = undirected$edge))
    score.i <- logLik(bn.dag, dm)
    if (score.i > score.best){
      score.best <- score.i
      select.id.best <- i
    }
  }
  max_ind = which.max(score_mat)
  best_model <- as.bn(pdag.set.direction(graph.AM = graph.AM, select.id = max_ind, undirected.edge = undirected$edge))
  return(best_model)
}








