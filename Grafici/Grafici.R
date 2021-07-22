library(dplyr)
library(ggplot2)
library(reshape2)
require(gridExtra)
require(readr)

#Percorsi in cui ci sono i file con le misurazioni
serial_M <- read_csv("Z:/CUDA/Progetto/Misure/serial_M.csv")
serial_L <- read_csv("Z:/CUDA/Progetto/Misure/serial_L.csv")
serial_XL <- read_csv("Z:/CUDA/Progetto/Misure/serial_XL.csv")

cudaonly_M <- read_csv("Z:/CUDA/Progetto/Misure/cudasync_M.csv")
cudaonly_L <- read_csv("Z:/CUDA/Progetto/Misure/cudasync_L.csv")
cudaonly_XL <- read_csv("Z:/CUDA/Progetto/Misure/cudasync_XL.csv")

openmp_M <- read_csv("Z:/CUDA/Progetto/Misure/openmp_M.csv")
openmp_L <- read_csv("Z:/CUDA/Progetto/Misure/openmp_L.csv")
openmp_XL <- read_csv("Z:/CUDA/Progetto/Misure/openmp_XL.csv")

openmpcuda_M <- read_csv("Z:/CUDA/Progetto/Misure/openmpcuda_M.csv")
openmpcuda_L <- read_csv("Z:/CUDA/Progetto/Misure/openmpcuda_L.csv")
openmpcuda_XL <- read_csv("Z:/CUDA/Progetto/Misure/openmpcuda_XL.csv")

alldata = rbind(
  serial_M, serial_L, serial_XL, 
  cudaonly_M, cudaonly_L, cudaonly_XL,
  openmp_M, openmp_L, openmp_XL,
  openmpcuda_M, openmpcuda_L, openmpcuda_XL
)

#Assegnamento delle categorie alle immagini
alldata <- alldata %>%
  mutate("category" = case_when(
    width == 2048 ~ "M",
    width == 4096 ~ "L",
    width == 8192 ~ "XL",
  ))
alldata$category <-  factor(alldata$category, levels = c("M", "L", "XL"))
alldata$algorithm <-  factor(alldata$algorithm, levels = c("Seriale", "OpenMP", "CUDA", "OpenMP + CUDA async"))


alldata$total = rowSums(alldata %>% select(
  "reading", 
  "processing",
  "writing"))

#fare la media di tutti quelli con lo stesso algoritmo
grouped_data <- alldata %>%
  group_by(category, algorithm) %>%
  summarise_at(vars("reading", "processing", "writing", "total"), funs(mean(., na.rm=TRUE)))

reading <- ggplot(grouped_data, aes(x=category, y=reading, fill=algorithm)) + 
  geom_bar(position = "dodge", stat="identity") + 
  xlab("Categoria") +
  ylab("Tempo di lettura (ms)") + 
  scale_fill_discrete(name = "Algoritmo") + 
  theme( legend.title = element_text(color = "black", size = 15),  
         legend.text = element_text(color = "black", size = 13),
         legend.position="bottom")
processing <- ggplot(grouped_data, aes(x=category, y=processing, fill=algorithm)) + 
  geom_bar(position = "dodge", stat="identity") +
  xlab("Categoria") +
  ylab("Tempo di generazione mipmap (ms)") + 
  scale_fill_discrete(name = "Algoritmo") +
  scale_y_continuous(trans = 'log10') + 
  theme( legend.title = element_text(color = "black", size = 15),  
         legend.text = element_text(color = "black", size = 13),
         legend.position="bottom")
writing <- ggplot(grouped_data, aes(x=category, y=writing, fill=algorithm)) + 
  geom_bar(position = "dodge", stat="identity") + 
  xlab("Categoria") +
  ylab("Tempo di scrittura (ms)") + 
  scale_fill_discrete(name = "Algoritmo") + 
  theme( legend.title = element_text(color = "black", size = 15),  
         legend.text = element_text(color = "black", size = 13),
         legend.position="bottom")

total <- ggplot(grouped_data, aes(x=category, y=total, fill=algorithm)) + 
  geom_bar(position = "dodge", stat="identity") + 
  xlab("Categoria") +
  ylab("Tempo totale (ms)") + 
  scale_fill_discrete(name = "Algoritmo") + 
  theme( legend.title = element_text(color = "black", size = 15),  
         legend.text = element_text(color = "black", size = 13),
         legend.position="bottom")


reading
processing
writing
total
