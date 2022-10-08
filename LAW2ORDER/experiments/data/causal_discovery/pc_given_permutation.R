suppressPackageStartupMessages({
  library(bnlearn)
})

pc_given_permutation <- function(dm.bn, permutation) {
  nodeNames <- colnames(dm.bn)
  nNodes <- length(permutation)
  blackList <- data.frame(matrix(data = "", nrow = nNodes * (nNodes - 1) / 2, ncol = 2), stringsAsFactors = FALSE)
  colnames(blackList) <- c("from", "to")
  cnt <- 1
  for (i in 1:(nNodes-1)) {
    for (j in (i+1):nNodes){
      blackList[cnt, ] <- nodeNames[permutation[c(j, i)]]
      cnt <- cnt + 1
    }
  }
  g.bn <- bnlearn::pc.stable(x = dm.bn, blacklist = blackList, alpha = 0.05)
  g.am <- as.graphAM(g.bn)
  adjMat <- as.data.frame(g.am@adjMat)
  rownames(adjMat) <- colnames(adjMat)
  return(adjMat)
}