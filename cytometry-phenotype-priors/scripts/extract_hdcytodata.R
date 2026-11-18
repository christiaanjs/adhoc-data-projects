# Extract expression matrix + row/col metadata from HDCytoData SummarizedExperiment .rda
# files WITHOUT needing Bioconductor installed (reads S4 slots via attributes()).
# Usage: Rscript extract_hdcytodata.R <in.rda> <out_dir>
args <- commandArgs(TRUE); dir.create(args[2], showWarnings = FALSE, recursive = TRUE)
objname <- suppressWarnings(load(args[1]))
o <- get(objname[1]); a <- attributes(o)
env <- attributes(a$assays)$.xData
m <- attributes(get("data", env))$listData$exprs
storage.mode(m) <- "double"
con <- file(file.path(args[2], "exprs.f64"), "wb"); writeBin(as.vector(m), con); close(con)  # column-major
writeLines(as.character(dim(m)), file.path(args[2], "dim.txt"))
rd <- as.data.frame(lapply(attributes(a$elementMetadata)$listData, as.character), stringsAsFactors = FALSE)
write.csv(rd, gzfile(file.path(args[2], "row_data.csv.gz")), row.names = FALSE)
cd <- as.data.frame(lapply(attributes(a$colData)$listData, as.character), stringsAsFactors = FALSE)
write.csv(cd, file.path(args[2], "col_data.csv"), row.names = FALSE)
md <- a$metadata
capture.output(str(md), file = file.path(args[2], "metadata_str.txt"))
cat(args[1], dim(m), "\n")
