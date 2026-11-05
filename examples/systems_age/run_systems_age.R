# 用法: Rscript run_systems_age.R SystemsAge_data.qs2 OUT.csv SAMPLE.csv [SAMPLE.csv ...]
# 样本 CSV: 行=样本, 列=CpG, 首列为样本名 (methylCIPHER 要求 样本x探针)
suppressMessages({library(methylCIPHER); library(data.table)})
a <- commandArgs(TRUE)
if (length(a) < 3) stop("用法: Rscript run_systems_age.R DATA.qs2 OUT.csv SAMPLE.csv [SAMPLE.csv ...]")
RData <- load_SystemsAge_data(a[1])
need <- names(RData$imputeMissingCpGs)
res <- list()
for (f in a[-(1:2)]) {
  d <- as.data.frame(fread(f)); rownames(d) <- d[[1]]; d <- as.matrix(d[, -1])
  cat(f, ": 命中 ", sum(need %in% colnames(d)), "/", length(need), "\n")
  r <- calcSystemsAge(d, pheno = data.frame(sample = rownames(d), file = basename(f)), RData = RData)
  res[[f]] <- r
}
out <- do.call(rbind, res)
write.csv(out, a[2], row.names = FALSE)
print(out)
