library("lme4")
library("car")
library("ggplot2")
data <-read.csv(file= 'burst_summary_per_trial_sum.csv')

glm<-glmer(burst_per_trial ~ epoch_type + (1 | subject), data=data, family=poisson(link="log"),control = glmerControl(optimizer="bobyqa",optCtrl=list(maxfun=2e6)))
Anova(glm)

