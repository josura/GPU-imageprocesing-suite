library(ggplot2)

times <-  read.csv("runningTimesMorph.csv")

times.small <- times[times$width == 1920,]

times.medium <- times[times$width == 2560,]

times.large <- times[times$width == 3840,]

ggplot(times.small,aes(x=method,y=runningTime,color=platform)) + 
  geom_boxplot(varwidth = TRUE) + 
  theme(axis.text.x = element_text(angle=90,vjust=0.5,hjust=1))

ggplot(times.medium,aes(x=method,y=runningTime,color=platform)) + 
  geom_boxplot(varwidth = TRUE)+ 
  theme(axis.text.x = element_text(angle=90,vjust=0.5,hjust=1))


ggplot(times.large,aes(x=method,y=runningTime,color=platform)) + 
  geom_boxplot(varwidth = TRUE)+ 
  theme(axis.text.x = element_text(angle=90,vjust=0.5,hjust=1))

timesDithering <-  read.csv("runningTimesDithering.csv")

timesDithering.small <- timesDithering[timesDithering$width == 1920,]

timesDithering.medium <- timesDithering[timesDithering$width == 2560,]

timetimesDithering.large <- timesDithering[timesDithering$width == 3840,]

ggplot(timesDithering.small,aes(x=method,y=runningTime,color=platform)) + 
  geom_boxplot(varwidth = TRUE) + 
  theme(axis.text.x = element_text(angle=90,vjust=0.5,hjust=1))

ggplot(timesDithering.medium,aes(x=method,y=runningTime,color=platform)) + 
  geom_boxplot(varwidth = TRUE)+ 
  theme(axis.text.x = element_text(angle=90,vjust=0.5,hjust=1))


ggplot(timetimesDithering.large,aes(x=method,y=runningTime,color=platform)) + 
  geom_boxplot(varwidth = TRUE)+ 
  theme(axis.text.x = element_text(angle=90,vjust=0.5,hjust=1))

timesSeg <-  read.csv("runningTimesSegmentation.csv")

timesSeg.small <- timesSeg[timesSeg$width == 1920,]

timesSeg.medium <- timesSeg[timesSeg$width == 2560,]

timesSeg.large <- timesSeg[timesSeg$width == 3840,]

ggplot(timesSeg.small,aes(x=method,y=runningTime,color=platform)) + 
  geom_boxplot(varwidth = TRUE) + 
  theme(axis.text.x = element_text(angle=90,vjust=0.5,hjust=1))

ggplot(timesSeg.medium,aes(x=method,y=runningTime,color=platform)) + 
  geom_boxplot(varwidth = TRUE)+ 
  theme(axis.text.x = element_text(angle=90,vjust=0.5,hjust=1))


ggplot(timesSeg.large,aes(x=method,y=runningTime,color=platform)) + 
  geom_boxplot(varwidth = TRUE)+ 
  theme(axis.text.x = element_text(angle=90,vjust=0.5,hjust=1))
