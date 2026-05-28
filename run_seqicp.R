library(seqICP)

getwd()

df <- read.table(
  file.choose(),
  sep = ";",
  header = TRUE
)


df <- na.omit(df)

timestamp_col <- "timestamp"
target_col <- "likelihood"


df[[timestamp_col]] <- NULL

Y <- df[[target_col]]
X <- df[, setdiff(names(df), target_col)]

X <- as.matrix(X)
Y <- as.numeric(Y)

rm(result)
gc()

n <- 3000
X_small <- X[1:n, ]
Y_small <- Y[1:n]

# quitar columnas con varianza cero o casi cero
vars <- apply(X_small, 2, var, na.rm = TRUE)
X_clean <- X_small[, vars > 1e-12, drop = FALSE]

# quitar columnas duplicadas
X_clean <- X_clean[, !duplicated(as.data.frame(X_clean)), drop = FALSE]

# escalar variables
X_clean <- scale(X_clean)

# centrar/escalar Y
Y_clean <- as.numeric(scale(Y_small))

dim(X_clean)

n_clean <- nrow(X_clean)

result_ar <- seqICP(
  X = X_clean,
  Y = Y_clean,
  test = "decoupled",
  par.test = list(
    grid = c(0, round(n_clean / 2), n_clean - 1),
    complements = FALSE,
    link = sum,
    alpha = 0.1,
    B = 50,
    permutation = FALSE
  ),
  model = "ar",
  par.model = list(
    pknown = TRUE,
    p = 1,
    max.p = 1
  ),
  max.parents = 2,
  stopIfEmpty = FALSE,
  silent = FALSE
)

result_ar$parent.set
result_ar$p.values
result_ar$modelReject

result_ar2 <- seqICP(
  X = X_clean,
  Y = Y_clean,
  test = "block.decoupled",
  par.test = list(
    grid = round(seq(0, n_clean - 1, length.out = 6)),
    complements = FALSE,
    link = sum,
    alpha = 0.05,
    B = 100,
    permutation = FALSE
  ),
  model = "ar",
  par.model = list(pknown = TRUE, p = 1, max.p = 1),
  max.parents = 1,
  stopIfEmpty = FALSE,
  silent = FALSE
)
result_ar2$modelReject
