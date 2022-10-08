library(SCCI)

id2vec <- function(id, vecRange){
  vecLen <- length(vecRange)
  vec <- vector(mode = 'integer', length = vecLen)
  divider <- c(1, cumprod(vecRange[1:(vecLen-1)]))
  remainder <- id - 1
  for (i in vecLen:2){
    vec[i] <- remainder %/% divider[i]
    remainder <- remainder %% divider[i]
  }
  vec[1] <- remainder
  return(vec + 1)
}

nSubForest <- function(g.am) {
  networkAdjMat <- g.am@adjMat
  nodeNames <- colnames(networkAdjMat)
  nVars <- dim(networkAdjMat)[1]
  if (any(networkAdjMat + t(networkAdjMat) == 2)){
    stop("DAG should be given not PDAG")
  }
  rawParentCnt <- colSums(networkAdjMat)
  hasMultipleParent <- rawParentCnt > 1
  multipleParentCnt <- rawParentCnt[hasMultipleParent]
  nForest <- prod(multipleParentCnt)
}

subBayesianForest <- function(g.am, bf.id){
  # in adjmat src\dst
  networkAdjMat <- g.am@adjMat
  nodeNames <- colnames(networkAdjMat)
  nVars <- dim(networkAdjMat)[1]
  rawParentCnt <- colSums(networkAdjMat)
  hasMultipleParent <- rawParentCnt > 1
  multipleParentCnt <- rawParentCnt[hasMultipleParent]
  nForest <- nSubForest(g.am)
  bf.id <- bf.id %/% nForest
  chosenParentVec <- id2vec(id = bf.id, vecRange = multipleParentCnt)
  forestAdjMat <- matrix(data = 0, nrow = nVars, ncol = nVars)
  colnames(forestAdjMat) <- colnames(networkAdjMat)
  rownames(forestAdjMat) <- rownames(networkAdjMat)
  chosenParentVecInd <- 1
  for (i in 1:nVars){
    if (hasMultipleParent[i]) {
      chosenParent <- chosenParentVec[chosenParentVecInd]
      forestAdjMat[which(networkAdjMat[, nodeNames[i]] == 1)[chosenParent], nodeNames[i]] <- 1
      chosenParentVecInd <- chosenParentVecInd + 1
    } else {
      forestAdjMat[, nodeNames[i]] <- networkAdjMat[, nodeNames[i]]
    }
  }
  forestAM <- graph::graphAM(adjMat = forestAdjMat, edgemode = 'directed')
  return(forestAM)
}

randomDirection <- function(g.am) {
  adjMat <- g.am@adjMat
  nVars <- dim(adjMat)[1]
  for (i in 1:(nVars-1)) {
    for (j in (i+1):nVars) {
      if (adjMat[i, j] == 1 && adjMat[j, i] == 1){
        if (runif(1) > 0.5) {
          adjMat[i, j] <- 0
        } else {
          adjMat[j, i] <- 0
        }
      }
    }
  }
  randomDAG <- graph::graphAM(adjMat = adjMat, edgemode = 'directed')
  return(randomDAG)
}


# this is the same as SCCI::regret(n, k) * log(2)
NMLRegretMultinomial <- function(n, k){
  cTable <- matrix(data = 1, nrow = n, ncol = 1)
  r1 <- 0:n
  r2 <- n:0
  cTable[2, 1] <- sum(choose(n = n, k = r1) * (r1 / n) ^ r1 * (r2 / n) ^ r2)
  for (i in 3:k) {
    cTable[i, 1] <- cTable[i-1, 1] + n / (i - 2) * cTable[i-2, 1]
  }
  return(log(cTable[k, 1]))
}


nodeDepth <- function(bf.am){
  adjMat <- bf.am@adjMat
  nodeNames <- names(adjMat)
  depthMat <- matrix(data = 0, nrow = 1, ncol = dim(adjMat)[1])
  names(depth) <- nodeNames
  for (nodeName in nodeNames){
    currNode <- nodeName
    hasParent <- TRUE
    depth <- 0
    while (hasParent) {
      parentInd <- which(adjMat[, currNode] == 1)
      if (length(parentInd) == 0){
        hasParent <- FALSE
      } else {
        currNode <- nodeNames[parentInd]
        depth <- depth + 1
      }
    }
    depthMat[currNode] <- depth
  }
  return(depthMat)
}


NMLRegretBayesianForest <- function(bf.am, dm){
  adjMat <- bf.am@adjMat
  nodeNames <- colnames(adjMat)
  dataFreq <- list()
  for (node in nodeNames) {
    dataFreq[[node]] <- table(dm[, node])
  }
  isLeaf <- rowSums(adjMat) == 0
  isRoot <- colSums(adjMat) == 0
  nData <- dim(bf.dm)[1]
  nVars <- dim(adjMat)[1]
  leafKi <- c()
  for (i in 1:nVars){
    if (isLeaf[i])
      leafKi <- c(leafKi, length(levels(bf.am[, i])))
  }
  kMax <- max(leafKi)
  CMN <- matrix(data = 0, nrow = kMax, ncol = nData)
  for (k in 1:kMax) {
    for (n in 1:nData)
      CMN[k, n] <- exp(SCCI::regret(n = n, k = k) * log(2))
  }
  depthMat <- nodeDepth(bf.am)
  maxDepth <- max(depthMat)
  bottomUpOrder <- c()
  for (d in depthMat:0){
    bottomUpOrder <- c(bottomUpOrder, which(maxDepth == d))
  }
  for (node in bottomUpOrder) {
    nCat <- length(dataFreq[[nodeNames[node]]])
    if (isLeaf[node]) {
      Lifpa <- 1
      parent <- which(adjMat[, node])
      parentFreq <- dataFreq[[nodeNames[parent]]]
      for (f in dataFreq[[nodeNames[node]]])
        Lifpa <- Lifpa * CMN[nCat, f]
    } else (!isRoot[node]) {
      children <- which(adjMat[node, ])
      parent <- which(adjMat[, node])
    } else {
      nodeParentFreq <- table(dm.bn[, c(node, parent)])
      freqRho <- rowSums(nodeParentFreq)
      freqGamma <- colSums(nodeParentFreq)
    }
  }
}

