# Daniel Varivoda
# Predicting NBA Rookie Impact



library(dplyr)
library(stringr)
library(ggplot2)
library(nbastatR)
library(xgboost)

library(logisticPCA)
library(caret)

#college player stats for 2008 - 2020
college_stats <- read.csv("trank_data.csv")
colnames(college_stats)

# [1] "player_name"  "team"         "conf"         "GP"           "Min_per"      "ORtg"         "usg"          "eFG"          "TS_per"      
# [10] "ORB_per"      "DRB_per"      "AST_per"      "TO_per"       "FTM"          "FTA"          "FT_per"       "twoPM"        "twoPA"       
# [19] "twoP_per"     "TPM"          "TPA"          "TP_per"       "blk_per"      "stl_per"      "ftr"          "yr"           "ht"          
# [28] "num"          "porpag"       "adjoe"        "pfr"          "year"         "pid"          "type"         "Rec_Rank"     "ast_tov"     
# [37] "rimmade"      "rimmade_ri"   "midmade"      "midmade_m"    "rimmade_ri.1" "midmade_m.1"  "dunksmade"    "dunksmiss_"   "dunksmade_"  
# [46] "pick"         "drtg"         "adrtg"        "dporpag"      "stops"        "bpm"          "obpm"         "dbpm"         "gbpm"        
# [55] "mp"           "ogbpm"        "dgbpm"        "oreb"         "dreb"         "treb"         "ast"          "stl"          "blk"         
# [64] "pts"          "Postion"


# getting each players stats for the year they were drafted -->  ideally would want to use all years
# viewing for improvement but currently utilizing how they were doing when drafted
stats_at_draft_yr <- college_stats %>% group_by(player_name) %>% slice_max(year) %>% ungroup()

## Creating the data frame we will work with.

all_stat <- bref_players_stats(
  seasons = c(2007:2020),
  tables = c("advanced"),
  include_all_nba = F,
  only_totals = FALSE,
  nest_data = FALSE,
  assign_to_environment = TRUE,
  widen_data = TRUE,
  join_data = TRUE,
  return_message = TRUE
)

#splitting the season string into an integer year for later joins
# e.g. (2017-18 --> 2017)
cc       <- strsplit(all_stat$slugSeason,'-')
part1    <- unlist(cc)[2*(1:length(all_stat$slugSeason))-1]

all_stat$slugSeason <- part1

# find which players were on multiple teams for the season
mult_team <- all_stat %>% group_by(slugPlayerBREF,slugSeason) %>% summarize("num_teams" = length(slugTeamBREF))
mult_team <- mult_team[which(mult_team$num_teams > 1),]

# remove the individual team statistics for players with multiple teams in a season to keep only aggregate
for(i in 1:nrow(mult_team)){
  cur_year <- mult_team$slugSeason[i]
  cur_player <- mult_team$slugPlayerBREF[i]
  all_stat <- all_stat[-which(all_stat$slugPlayerBREF == cur_player &
                                all_stat$slugSeason == cur_year &
                                all_stat$slugTeamBREF != "TOT"),]
  
}

#what cols do we need (VORP, BPM)
all_stat <- all_stat[,c(1,2,3,8,11,39,40)]

all_stat <- all_stat[which(all_stat$countGames > 20),]



#function to scale values from 0-1 for standardizing using provided min, max
range01 <- function(x,m_min, m_max){(x-m_min)/(m_max-m_min)}


# find the min and max for the target statistics for each season to scale
min_max_by_szn <- all_stat %>% group_by(slugSeason) %>% summarize("min_BPM" = min(ratioBPM), "max_BPM" = max(ratioBPM),
                                                                  "min_VORP" = min(ratioVORP), "max_VORP" = max(ratioVORP))

#join the min max stats to their respective seasons on the df
all_stat <- all_stat %>% left_join(min_max_by_szn, by = c("slugSeason" = "slugSeason"))

summary(all_stat$min_VORP)

# Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# -2.000  -1.500  -1.400  -1.402  -1.200  -0.900 

#normalize the VORP and BPM measurements
std_VORP <- range01(all_stat$ratioVORP, all_stat$min_VORP, all_stat$max_VORP)
std_BPM <- range01(all_stat$ratioBPM, all_stat$min_BPM, all_stat$max_BPM)

all_stat <- cbind(all_stat, std_VORP, std_BPM)

summary(std_VORP)
summary(std_BPM)

# > summary(std_VORP)
# Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# 0.0000  0.1215  0.1696  0.2074  0.2544  1.0000 
# > summary(std_BPM)
# Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# 0.0000  0.3448  0.4308  0.4365  0.5212  1.0000 


#read in the NBA.com data
all_Stars_19 <- read.csv("ASG_19.csv")
players_2019 <- read.csv("ASG_to_predict.csv")

players_2019$Selected. <- 0
players_2019$Selected.[which(players_2019$PLAYER %in% all_Stars_19$name)] <- 1

players_pre_2019 <- read.csv("ASG_train.csv")


players_total <- rbind(players_pre_2019, players_2019)
ncol(players_total)
# [1] 22

#outliers in PIE that stem from players not playing games
quantile(players_total$PIE)

#   0%    25%    50%    75%   100% 
# -400.0    6.1    8.6   11.0  300.0 

#only including players that played at least 1/4 of the teams games
players_total$percent_played <- players_total$GP/players_total$Team.GP
players_total <- players_total[which(players_total$percent_played >= .25),]

summary(players_total$PIE)
# 
# Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# -16.700   6.500   8.800   8.769  11.000  28.300

#much better now getting ready to normalize our PIE
min_max_PIE_szn <- players_total %>% group_by(Year) %>% summarize("min_PIE" = min(PIE), "max_PIE" = max(PIE))

players_total <- players_total %>% left_join(min_max_PIE_szn, by = c("Year" = "Year"))
std_PIE <- range01(players_total$PIE, players_total$min_PIE, players_total$max_PIE)

players_total$std_PIE <- std_PIE
summary(std_PIE)
# now in [0,1] range
# Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# 0.0000  0.4062  0.5155  0.5092  0.6151  1.0000 

# NOW JOIN DFS TO MAKE COMPOSITE MEASURE (FIRST NEED TO MAKE YEAR AN INT ON ALL_STAT)
all_stat$slugSeason <- as.numeric(all_stat$slugSeason)

stats_at_final_yr$college_year[which(stats_at_final_yr$player_name == "Blake Griffin")]
unique(stats_at_final_yr$college_year)

#only recent players needed
players_total <- players_total[which(players_total$Year > 2005),]


final_stat <- all_stat %>% inner_join(players_total[,c(1,3,26)], by = c("namePlayer"="PLAYER", "slugSeason" = "Year"))

# average of 3 normalized metric for the composite metric
final_stat$composite_mes <- rowMeans(cbind(final_stat$std_VORP, final_stat$std_BPM, final_stat$std_PIE))

summary(final_stat$composite_mes)

# Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# 0.02963 0.29993 0.37095 0.38628 0.45446 1.00000 

#dictionary of players found on bballref
player_dict <- dictionary_bref_players()
player_dict <- player_dict[-which(is.na(player_dict$slugSeasonRookie)),]

# similar to above, split the string to find an int year for the rookie season
rook      <- strsplit(player_dict$slugSeasonRookie,'-')
rk_y    <- unlist(rook)[2*(1:length(player_dict$slugSeasonRookie))-1]

player_dict$slugSeasonRookie <- rk_y

#finding rookie season for each unique player to add later
player_dict  <- player_dict[,c(2,6)]

#get the median composite metric value by year to create the binary response
median_impact_by_year <- final_stat %>% group_by(slugSeason) %>% summarize("median_impact" = quantile(composite_mes)[3])

final_stat <- final_stat %>% left_join(median_impact_by_year, by = c("slugSeason" = "slugSeason"))
players_total <- final_stat
players_total$strong_impact <- 0 
players_total$strong_impact[which(players_total$composite_mes > players_total$median_impact)] <- 1

#add the players by rookie year to filter out players that are not in their rookie year
players_total <- players_total %>% left_join(player_dict, by = ("slugPlayerBREF" = "slugPlayerBREF"))
players_total$slugSeasonRookie <- as.numeric(players_total$slugSeasonRookie)

players_total_rookie_szn <- players_total[which(players_total$slugSeason == players_total$slugSeasonRookie),]


# keeping only relevant data for the nba players rookie seasons (identifiers, position, binary response)
new_p_t <- players_total_rookie_szn[,c(1,2,3,4,17)]


players_impact <- new_p_t %>% inner_join(stats_at_final_yr, by= c("namePlayer" = "player_name"))

nrow(players_impact)
#[1] 444




#find how many NAs present in each column of the data frame

nas_by_col <- c()
for(i in 1:ncol(players_impact)){
  col <- players_impact[,i]
  nas_by_col <- c(nas_by_col, length(which(is.na(col))))
}


names(nas_by_col) <- 1:length(nas_by_col)
nas_by_col

# 1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32  33  34  35  36  37  38 
# 0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   3  76  76  76  76 
# 39  40  41  42  43  44  45  46  47  48  49  50  51  52  53  54  55  56  57  58  59  60  61 
# 76  77  76  76 112   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0 


#remove columns with many NAs
players_impact <- players_impact[,-c(35:43)]

#convert height from character
players_impact$college_ht <- as.numeric(players_impact$college_ht)

# want players with at least 10 college games to get accurate stats
summary(players_impact$college_GP)

# Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# 3.00   32.00   34.00   33.03   36.00   41.00 

# number of players with under 10 games
nrow(players_impact[which(players_impact$college_GP < 10),])
#6

#stat discrepancy for low games played
summary(players_impact$college_ORtg[which(players_impact$college_GP < 10)])

# Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# 0.00   65.97   71.50   68.35   87.53  110.10 

players_impact <- players_impact[-which(players_impact$college_GP < 10),]

# creating binary matrix for categorical position
mat <- model.matrix(~ players_impact$groupPosition -1)
players_impact <- cbind(players_impact,mat)

nrow(players_impact)

#438


# creating a row with column  numbers to the df to make identifying which columns are which easier
num_col <- 1:ncol(players_impact)
my_mat <- data.frame("row_0" = 0)
for(i  in 1:length(num_col)){
  my_mat <- cbind(my_mat, i)
}
my_mat <- my_mat[,-1]
names(my_mat) <- names(players_impact)

#making the first row the column number to easily access which columns to remove
players_impact <- rbind(my_mat, players_impact)


#set4: remove gbpm keep ogbpm,dgbpm --> not needed 1,3,4,7,15,18,21,25,29,30,32,36,37,38,39,45

players_impact <- players_impact[-1,]

players_impact_4 <- players_impact[,-c(1,2,3,4,6,7,10,16,19,22,28,33,34,35,39,40,41,42,48,54,55)]
players_impact_4_name <- players_impact[,-c(1,3,4,6,7,10,16,19,22,28,33,34,35,39,40,41,42,48,54,55)]

#percent of strong impact players
nrow(players_impact_4[which(players_impact_4$strong_impact == 1),])/(nrow(players_impact_4[which(players_impact_4$strong_impact == 0),])+nrow(players_impact_4[which(players_impact_4$strong_impact == 1),]))
#[1] 0.2123288







# EDA

pca_impact <- players_impact_4

#only keep rows with no NA vlaues
pca_impact <- pca_impact[which(complete.cases(pca_impact)), ]
names <- ifelse(pca_impact$strong_impact < 1, "weak", "strong")


library(ggfortify)

df <- pca_impact[,-1]

pca_res <- prcomp(df, scale. = TRUE)
summary(pca_res)

#PCA % variation explained graph
fviz_eig(pca_res)
pca_impact$strong_impact <- as.factor(pca_impact$strong_impact)
#plot of pca by impact
autoplot(pca_res, data = pca_impact, colour = "strong_impact")


#creating a df to make it easier to graph for EDA
eda_df <- data.frame("Player_Impact_Estimate_Category" = rep("Low", nrow(players_impact)), players_impact)
eda_df$Player_Impact_Estimate_Category[which(eda_df$strong_impact == 1)] <- "High"
eda_df$Player_Impact_Estimate_Category <- as.factor(eda_df$Player_Impact_Estimate_Category)

# visualize number of high impact v slow impact players
ggplot(eda_df, aes(x=Player_Impact_Estimate_Category, fill = Player_Impact_Estimate_Category)) + 
  stat_count(width = 0.5)+ 
  scale_fill_manual("Player Impact Estimate",labels = c("Low", "High"), values = c("#F8766D","#619CFF")) +
  scale_x_discrete(breaks=c(0, 1),labels=c("Low", "High")) + labs(title = "Distribution of Impact Levels")



# creating a correlation matrix to see what variables are highly correlated with each other
res <- cor(eda_df[,-c(1,2,3,4,5,29)])
res <- round(res, 2)
res[1,]

# strong_impact                    college_GP               college_Min_per                  college_ORtg                   college_usg 
# 1.00                         -0.02                          0.00                          0.18                          0.03 
# college_eFG                college_TS_per               college_ORB_per               college_DRB_per               college_AST_per 
# 0.18                          0.18                          0.11                          0.13                          0.08 
# college_TO_per                   college_FTM                   college_FTA                college_FT_per                 college_twoPM 
# -0.01                          0.07                          0.07                          0.00                          0.09 
# college_twoPA              college_twoP_per                   college_TPM                   college_TPA                college_TP_per 
# 0.04                          0.19                         -0.09                         -0.10                         -0.07 
# college_blk_per               college_stl_per                   college_ftr                    college_ht                college_porpag 
# 0.18                          0.06                          0.11                          0.07                          0.14 
# college_adjoe                   college_pfr                  college_year               college_ast_tov                  college_drtg 
# 0.16                          0.01                         -0.11                          0.04                         -0.13 
# college_adrtg               college_dporpag                 college_stops                   college_bpm                  college_obpm 
# -0.12                          0.09                          0.08                          0.22                          0.14 
# college_dbpm                  college_gbpm                    college_mp                 college_ogbpm                 college_dgbpm 
# 0.18                          0.21                          0.02                          0.15                          0.18 
# college_oreb                  college_dreb                  college_treb                   college_ast                   college_stl 
# 0.09                          0.10                          0.10                          0.08                          0.07 
# college_blk                   college_pts players_impact.groupPositionC players_impact.groupPositionF players_impact.groupPositionG 
# 0.16                          0.05                          0.13                         -0.09                         -0.01 


library(dplyr)


# making it easier to see grade proportions
eda_df_year <- eda_df %>% group_by(Player_Impact_Estimate_Category,college_yr) %>% summarize("count" = length(college_yr))
high_tot <- sum(eda_df_year$count[which(eda_df_year$Player_Impact_Estimate_Category == "High")])
low_tot <- sum(eda_df_year$count[which(eda_df_year$Player_Impact_Estimate_Category == "Low")])
eda_df_year$prop <- c(eda_df_year$count[which(eda_df_year$Player_Impact_Estimate_Category == "High")]/high_tot,
                      eda_df_year$count[which(eda_df_year$Player_Impact_Estimate_Category == "Low")]/low_tot)

# df for proportions of each position grouping
eda_df_pos <- eda_df %>% group_by(Player_Impact_Estimate_Category,groupPosition) %>% summarize("count" = length(college_yr))
high_tot <- sum(eda_df_pos$count[which(eda_df_pos$Player_Impact_Estimate_Category == "High")])
low_tot <- sum(eda_df_pos$count[which(eda_df_pos$Player_Impact_Estimate_Category == "Low")])
eda_df_pos$prop <- c(eda_df_pos$count[which(eda_df_pos$Player_Impact_Estimate_Category == "High")]/high_tot,
                     eda_df_pos$count[which(eda_df_pos$Player_Impact_Estimate_Category == "Low")]/low_tot)

# plot of poition proportions
ggplot(eda_df_pos, aes(x = groupPosition, y = prop, fill = Player_Impact_Estimate_Category)) + 
  geom_bar(stat = "identity", position = position_dodge()) +
  scale_y_continuous(labels = c("0%", "5%", "10%","15%","20%", "25%","30%" ,"35%","40%", "45%", "50%","55%", "60%"),
                     breaks = seq(0,.6,by = .05) ) +
  labs(title = "Player Position by Rookie Impact Estimate",
       x = "Player Position Group",
       y = "Proportion")

# plot of grade proportions
ggplot(eda_df_year, aes(x = college_yr, y = prop, fill = Player_Impact_Estimate_Category)) + 
  geom_bar(stat = "identity", position = position_dodge()) +
  scale_y_continuous(labels = c("0%", "5%", "10%","15%","20%", "25%","30%" ,"35%","40%"),
                     breaks = seq(0,.4,by = .05) ) +
  labs(title = "Player Grade Before Draft by Rookie Impact Estimate",
       x = "Player Grade",
       y = "Proportion")

#seems like one & done prospects (freshmen) may be a good measure of a players potential impact, though that is likely
# is because well-regarded players enter the draft earlier rather than because of the effect of being a freshman


#plot of height
ggplot(eda_df, aes(x = college_ht, fill = Player_Impact_Estimate_Category, color = Player_Impact_Estimate_Category)) + 
  geom_histogram(aes(y = ..density..), position = position_dodge(), binwidth = 2 ) + 
  geom_density(alpha = 0.2) +
  labs(title = "Player Height by NBA Rookie Year Impact Level",
       x = "Height (In)",
       y = "Density") + scale_color_brewer(palette = "Paired") + 
  scale_x_continuous(breaks = seq(50,100, by = 2))+
  theme_classic() + theme(legend.position = "top", plot.title = element_text(hjust = 0.5))

#plot of offensive rating
ggplot(eda_df, aes(x = college_ORtg, fill = Player_Impact_Estimate_Category, color = Player_Impact_Estimate_Category)) + 
  geom_histogram(aes(y = ..density..), position = position_dodge(), binwidth = 3 ) + 
  geom_density(alpha = 0.2) +
  labs(title = "College Offensive Rating by NBA Rookie Year Impact Level",
       x = "Offensive Rating",
       y = "Density") + scale_color_brewer(palette = "Paired") + 
  theme_classic() + theme(legend.position = "top", plot.title = element_text(hjust = 0.5))

#plot of true shooting
ggplot(eda_df, aes(x = college_TS_per, fill = Player_Impact_Estimate_Category, color = Player_Impact_Estimate_Category)) + 
  geom_histogram(aes(y = ..density..), position = position_dodge(), binwidth = 3 ) + 
  geom_density(alpha = 0.2) +
  labs(title = "College True Shooting by NBA Rookie Year Impact Level",
       x = "True Shooting",
       y = "Density") + scale_color_brewer(palette = "Paired") + 
  theme_classic() + theme(legend.position = "top", plot.title = element_text(hjust = 0.5))

#plot of defensive rating
ggplot(eda_df, aes(x = college_drtg, fill = Player_Impact_Estimate_Category, color = Player_Impact_Estimate_Category)) + 
  geom_histogram(aes(y = ..density..), position = position_dodge(), binwidth = 3 ) + 
  geom_density(alpha = 0.2) +
  labs(title = "College Defensive Rating by NBA Rookie Year Impact Level",
       x = "Defensive Rating",
       y = "Density") + scale_color_brewer(palette = "Paired") + 
  theme_classic() + theme(legend.position = "top", plot.title = element_text(hjust = 0.5))

#plot of points over average replacement adjusted
ggplot(eda_df, aes(x = college_porpag, fill = Player_Impact_Estimate_Category, color = Player_Impact_Estimate_Category)) + 
  geom_histogram(aes(y = ..density..), position = position_dodge(), binwidth = 1 ) + 
  geom_density(alpha = 0.2) +
  labs(title = "College Points Over Replacement Per Adjusted Game by Impact Level",
       x = "Points Over Replacement",
       y = "Density") + scale_color_brewer(palette = "Paired") + 
  theme_classic() + theme(legend.position = "top", plot.title = element_text(hjust = 0.5))

#plot of adjusted box plus/minues
ggplot(eda_df, aes(x = college_gbpm, fill = Player_Impact_Estimate_Category, color = Player_Impact_Estimate_Category)) + 
  geom_histogram(aes(y = ..density..), position = position_dodge(), binwidth = 2 ) + 
  geom_density(alpha = 0.2) +
  labs(title = "College Adjusted Box Plus/Minus by Impact Level",
       x = "Box Plus/Minus",
       y = "Density") + scale_color_brewer(palette = "Paired") + 
  theme_classic() + theme(legend.position = "top", plot.title = element_text(hjust = 0.5))

#plot of block percentage

ggplot(eda_df, aes(x = college_blk_per, fill = Player_Impact_Estimate_Category, color = Player_Impact_Estimate_Category)) + 
  geom_histogram(aes(y = ..density..), position = position_dodge(), binwidth = .5 ) + 
  geom_density(alpha = 0.2) +
  labs(title = "College Block Percentage by Impact Level",
       x = "Block Percentage",
       y = "Density") + scale_color_brewer(palette = "Paired") + 
  theme_classic() + theme(legend.position = "top", plot.title = element_text(hjust = 0.5))










library(xgboost)
library(caret)

# create labels
labels_players_impact_4 <- players_impact_4$strong_impact

# find the number of rows in the test df; 15% split
test_size <- round(nrow(players_impact_4) * .15)

#set seed to keep results consistent
set.seed(04101998)

#randomly slect rows to be in the test set
test_num <- sample(1:nrow(players_impact_4), test_size)

#create testing, training, and validation sets
test <- players_impact_4[test_num,]
train <- players_impact_4[-test_num,]

valid_size <- round(nrow(players_impact_4) * .15)
valid_num <- sample(1:nrow(train), valid_size)


valid <- train[valid_num,]
train <- train[-valid_num, ]

# labels for our subsets
label_test <- test[,1]
label_train <- train[,1]
label_valid <- valid[,1]

test <- test[,-1]
train <- train[,-1]
valid <- valid[,-1]


train <- as.matrix(train)
test <- as.matrix(test)
valid <- as.matrix(valid)

dtrain <- xgb.DMatrix(data = train,label = label_train) 
dtest <- xgb.DMatrix(data = test,label=label_test)
dvalid <- xgb.DMatrix(data = valid,label=label_valid)


#parameters for our future gradient boosted model
params <- list(booster = "gbtree", objective = "binary:logistic", 
               eta=0.3, gamma=0, max_depth=6, min_child_weight=1, subsample=1, colsample_bytree=1)

#cross validation to find the best stoppin iteration for the base model
xgbcv <- xgb.cv( params = params, data = dtrain, nrounds = 100, 
                 nfold = 10, showsd = T, stratified = T, 
                 print.every.n = 10, early.stop.round = 50, maximize = F)

# our best iteration was 4
xgbcv$best_iteration
#[1] 4

print(xgbcv, verbose=TRUE)

##### xgb.cv 10-folds
# call:
#   xgb.cv(params = params, data = dtrain, nrounds = 100, nfold = 10, 
#          showsd = T, stratified = T, maximize = F, print.every.n = 10, 
#          early.stop.round = 50)
# params (as set within xgb.cv):
#   booster = "gbtree", objective = "binary:logistic", eta = "0.3", gamma = "0", max_depth = "6", min_child_weight = "1", subsample = "1", colsample_bytree = "1", print_every_n = "10", early_stop_round = "50", silent = "1"
# callbacks:
#   cb.print.evaluation(period = print_every_n, showsd = showsd)
# cb.evaluation.log()
# cb.early.stop(stopping_rounds = early_stopping_rounds, maximize = maximize, 
#               verbose = verbose)
# niter: 54
# best_iteration: 4
# best_ntreelimit: 4
# evaluation_log:
#   iter train_logloss_mean train_logloss_std test_logloss_mean test_logloss_std
# 1          0.5512790      0.0084306868         0.6033244       0.02592626
# 2          0.4601047      0.0157455736         0.5551365       0.04194593
# 3          0.3890143      0.0176310000         0.5391823       0.06078864
# 4          0.3343502      0.0178696350         0.5293341       0.07714868
# 5          0.2868474      0.0145722363         0.5300931       0.08500419
# 6          0.2486557      0.0126645242         0.5352293       0.09642975
# 7          0.2163785      0.0101826949         0.5439931       0.10785461
# 8          0.1908077      0.0086750723         0.5438160       0.11876304
# 9          0.1682592      0.0077343556         0.5539567       0.12313287
# 10          0.1513894      0.0080012218         0.5572623       0.12319995
# 11          0.1368601      0.0063471177         0.5619369       0.12122252
# 12          0.1247699      0.0061032156         0.5678710       0.12282756
# 13          0.1144112      0.0061060803         0.5711988       0.12733095
# 14          0.1048486      0.0054468044         0.5816036       0.13542261
# 15          0.0968513      0.0051824072         0.5847044       0.13532859
# 16          0.0898714      0.0044592460         0.5922195       0.14090316
# 17          0.0839220      0.0045816156         0.5968153       0.14807287
# 18          0.0782662      0.0041008833         0.6039006       0.15339857
# 19          0.0735633      0.0038082609         0.6074316       0.15264115
# 20          0.0689968      0.0032041160         0.6111337       0.14862718
# 21          0.0654100      0.0029360380         0.6198208       0.14861281
# 22          0.0617965      0.0026797304         0.6272763       0.15577703
# 23          0.0586962      0.0023305474         0.6305202       0.16052440
# 24          0.0560511      0.0022805428         0.6393489       0.16350429
# 25          0.0535525      0.0022507035         0.6436480       0.16456494
# ...
# iter train_logloss_mean train_logloss_std test_logloss_mean test_logloss_std
# Best iteration:
#   iter train_logloss_mean train_logloss_std test_logloss_mean test_logloss_std
# 4          0.3343502        0.01786964         0.5293341       0.07714868


#train the model with base paramters
xgb1 <- xgb.train (params = params, data = dtrain, nrounds = 4, watchlist = list(val=dvalid,train=dtrain), print.every.n = 10, early.stop.round = 50, maximize = F , eval_metric = "error")
#model prediction
xgbpred <- predict (xgb1,dtest)
xgbpred <- ifelse (xgbpred > 0.3,1,0)


caret::confusionMatrix(factor(xgbpred), factor(label_test))
# Accuracy : 0.7273      


#importance matrix
mat <- xgb.importance (feature_names = colnames(train),model = xgb1)
print(mat)

# Confusion Matrix and Statistics
# 
# Reference
# Prediction  0  1
# 0 40 10
# 1  9  7
# 
# Accuracy : 0.7121         
# 95% CI : (0.5875, 0.817)
# No Information Rate : 0.7424         
# P-Value [Acc > NIR] : 0.7627         
# 
# Kappa : 0.2326         
# 
# Mcnemar's Test P-Value : 1.0000         
#                                          
#             Sensitivity : 0.8163         
#             Specificity : 0.4118         
#          Pos Pred Value : 0.8000         
#          Neg Pred Value : 0.4375         
#              Prevalence : 0.7424         
#          Detection Rate : 0.6061         
#    Detection Prevalence : 0.7576         
#       Balanced Accuracy : 0.6140         
#                                          
#        'Positive' Class : 0  

xgb.plot.importance (importance_matrix = mat[1:20]) 

#our decision tree for the base model
xgb.plot.tree(model = xgb1)



# Creating the Tuned Model
library(mlr)
library(parallel)
library(parallelMap)

set.seed(04101998)

# Creating training, validation, and test dfs
names(players_impact_4)[34] <- "Center"
players_impact_4_2 <- players_impact_4
players_impact_4_2$strong_impact <- as.factor(players_impact_4_2$strong_impact)
test_1 <- players_impact_4_2[test_num,]
train_1 <- players_impact_4_2[-test_num,]

valid_1 <- train_1[valid_num,]
train_1 <- train_1[-valid_num, ]

summary(test_1$strong_impact)
summary(train_1$strong_impact)

#create tasks
traintask <- makeClassifTask (data = train_1,target = "strong_impact")
testtask <- makeClassifTask (data = valid_1,target = "strong_impact")
testtask_2 <- makeClassifTask (data = test_1,target = "strong_impact")

#create learner
lrn <- makeLearner("classif.xgboost",predict.type = "prob")
lrn$par.vals <- list( objective="binary:logistic", eval_metric="error", nrounds=100L, eta=0.1)

#set parameter space
params <- makeParamSet( makeDiscreteParam("booster",values = c("gbtree")), makeIntegerParam("max_depth",lower = 3L,upper = 10L), makeNumericParam("min_child_weight",lower = 1L,upper = 10L), makeNumericParam("subsample",lower = 0.5,upper = 1), makeNumericParam("colsample_bytree",lower = 0.5,upper = 1))

#set resampling strategy
rdesc <- makeResampleDesc("CV",stratify = T,iters=5L)

#search strategy
ctrl <- makeTuneControlRandom()

parallelStartSocket(cpus = detectCores())

#parameter tuning
mytune <- tuneParams(learner = lrn, task = traintask, resampling = rdesc, measures = acc, par.set = params, control = ctrl, show.info = T)

#tuner accuracy
mytune$y
# acc.test.mean 
# 0.8104178 

#tuned parameters
mytune$x

# $booster
# [1] "gbtree"
# 
# $max_depth
# [1] 3
# 
# $min_child_weight
# [1] 6.457339
# 
# $subsample
# [1] 0.5913547
# 
# $colsample_bytree
# [1] 0.5102951


#set the new parameters for the model
lrn_tune <- setHyperPars(lrn,par.vals = mytune$x)

#train the tuned model
xgmodel <- mlr::train(learner = lrn_tune,task = traintask)

#predicting the test set
xgpred <- predict(xgmodel,testtask_2)
xgpred <- setThreshold(xgpred, .75)

caret::confusionMatrix(xgpred$data$response,xgpred$data$truth)
# Confusion Matrix and Statistics
# 
# Reference
# Prediction  0  1
# 0 44  9
# 1  5  8
# 
# Accuracy : 0.7879          
# 95% CI : (0.6698, 0.8789)
# No Information Rate : 0.7424          
# P-Value [Acc > NIR] : 0.2446          
# 
# Kappa : 0.3992          
# 
# Mcnemar's Test P-Value : 0.4227          
#                                           
#             Sensitivity : 0.8980          
#             Specificity : 0.4706          
#          Pos Pred Value : 0.8302          
#          Neg Pred Value : 0.6154          
#              Prevalence : 0.7424          
#          Detection Rate : 0.6667          
#    Detection Prevalence : 0.8030          
#       Balanced Accuracy : 0.6843          
#                                           
#        'Positive' Class : 0 

#tuned model feature importance
getFeatureImportance(xgmodel)

# FeatureImportance:
#   Task: train_1
# 
# Learner: classif.xgboost
# Measure: NA
# Contrast: NA
# Aggregation: function (x)  x
# Replace: NA
# Number of Monte-Carlo iterations: NA
# Local: FALSE
# # A tibble: 6 x 2
# variable        importance
# <chr>                <dbl>
#   1 college_ORtg        0.0318
# 2 college_usg         0.0442
# 3 college_TS_per      0.0425
# 4 college_ORB_per     0.0132
# 5 college_DRB_per     0.0265
# 6 college_AST_per     0.0538
