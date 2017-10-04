library(ggplot2)


read.csv("network_data.csv", header=FALSE) %>%
    rename(m=V1, d=V2, num_of_layers=V3, num_of_nodes=V4) %>%
    ggplot(aes(x=num_of_nodes, y=avgLoss, group=m, color=m)) + geom_boxplot()
