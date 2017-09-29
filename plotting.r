library(rgl)
library(dplyr)
mat1 <- matrix(c(.7, -.7, 0, 0,
               .3, .3, .9, 0,
               -.7, -.7, .4, 0,
               0, 0, 0, 1), nrow=4, ncol=4, byrow=TRUE)

col <-  rainbow(100)
color_pallet <- function(x){
    for(i in 1:99){
        if(y[[i+1]][1]>x & x>=y[[i]][1]){
            return(col[i])
        }
    }
    return(col[100])
}

data <- read.table("model_data.csv") %>%
    rename(x = V1, y = V2, v = V3)
v.min <- min(data$v)
v.max <- max(data$v)
v.split <- split(data$v[order(data$v)], ceiling(seq_along(data$v)/(length(data$v)/100)))
y <- v.split
col <-  rainbow(100)
color_pallet <- function(x){
    for(i in 1:99){
        if(y[[i+1]][1]>x & x>=y[[i]][1]){
            return(col[i])
        }
    }
    return(col[100])
}
v.colors <- unlist(sapply(data$v, color_pallet))

plot3d(data$x, data$y, data$v, type="p", col=v.colors,
           zlim=c(v.min, v.max), box=FALSE, xlab="", ylab="", zlab="", lit=FALSE)
    rgl.viewpoint(scale=c(1, 1, 17/(v.max-v.min)), userMatrix=mat1)
