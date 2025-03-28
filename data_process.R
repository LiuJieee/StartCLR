library(stringr)
# 加载突变序列数据
seq <- read.csv("./dataset/sample_file.txt", sep="\t", header=T)

# HyenaDNA输入
data_alt <- seq[,7]
write.table(as.data.frame(data_alt), file="./dataset/sample_HyenaDNA_input.txt",
            sep="\t", row.names=F, col.names=F, quote=F)

#############################################################################################

# GPNMSA输入
###原长1001bp的序列，突变位点位于第501个位置，向前截63个碱基，向后截64个碱基，共长128bp
seq[8] <- str_sub(seq[,7], 438, 565)
str_sub(seq[,8], 64, 64) # 检查第64个碱基是否与ALT列碱基一致
names(seq)[8] <- "MutSite_64_128seq"


###将截取的128bp序列进行字符转换，A为1，C为2，G为3，T为4
seq[, 8] <- sapply(seq[, 8], function(x) {
  # 替换字符
  x <- gsub("A", "1", x)
  x <- gsub("C", "2", x)
  x <- gsub("G", "3", x)
  x <- gsub("T", "4", x)
  return(x)
})

###只保留需要的信息
data_pos <- seq[,c(2,3,8)]

# 将第三列的字符串按字符拆分
split_data <- do.call(rbind, strsplit(as.character(data_pos[, 3]), ""))

# 将拆分后的数据框转换为数据框并添加列名
split_data <- as.data.frame(split_data, stringsAsFactors = FALSE)

# 如果需要，可以将拆分后的数据框合并回原数据框
data1 <- cbind(data_pos[,1:2], split_data)

write.table(data1, "./dataset/sample_GPNMSA_input.csv", row.names = FALSE, col.names=F, sep=",", quote = FALSE)