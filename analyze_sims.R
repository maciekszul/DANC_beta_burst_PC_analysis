library(ggplot2)
library(lme4)
library(car)

code.poly <- function(df=NULL, predictor=NULL, poly.order=NULL, orthogonal=TRUE, draw.poly=TRUE){
  require(reshape2)
  require(ggplot2)
  
  # convert choice for orthogonal into choice for raw
  raw <- (orthogonal-1)^2
  
  # make sure that the declared predictor is actually present in the data.frame
  if (!predictor %in% names(df)){
    warning(paste0(predictor, " is not a variable in your data frame. Check spelling and try again"))
  }
  
  # Extract the vector to be used as the predictor
  predictor.vector <- df[,which(colnames(df)==predictor)]
  
  # create index of predictor (e.g. numbered time bins)
  # the index of the time bin will be used later as an index to call the time sample
  predictor.indices <- as.numeric(as.factor(predictor.vector))
  
  df$temp.predictor.index <- predictor.indices
  
  #create x-order order polys (orthogonal if not raw)
  predictor.polynomial <- poly(x = unique(sort(predictor.vector)),
                               degree = poly.order, raw=raw)
  
  # use predictor index as index to align
  # polynomial-transformed predictor values with original dataset
  # (as many as called for by the polynomial order)
  df[, paste("poly", 1:poly.order, sep="")] <-
    predictor.polynomial[predictor.indices, 1:poly.order]
  
  # draw a plot of the polynomial transformations, if desired
  if (draw.poly == TRUE){
    # extract the polynomials from the df
    df.poly <- unique(df[c(predictor, paste("poly", 1:poly.order, sep=""))])
    
    # melt from wide to long format
    df.poly.melt <- melt(df.poly, id.vars=predictor)
    
    # Make level names intuitive
    # don't bother with anything above 6th order.
    levels(df.poly.melt$variable)[levels(df.poly.melt$variable)=="poly1"] <- "Linear"
    levels(df.poly.melt$variable)[levels(df.poly.melt$variable)=="poly2"] <- "Quadratic"
    levels(df.poly.melt$variable)[levels(df.poly.melt$variable)=="poly3"] <- "Cubic"
    levels(df.poly.melt$variable)[levels(df.poly.melt$variable)=="poly4"] <- "Quartic"
    levels(df.poly.melt$variable)[levels(df.poly.melt$variable)=="poly5"] <- "Quintic"
    levels(df.poly.melt$variable)[levels(df.poly.melt$variable)=="poly6"] <- "Sextic"
    
    # change some column names for the output
    colnames(df.poly.melt)[colnames(df.poly.melt) == "variable"] <- "Order"
    
    poly.plot <- ggplot(df.poly.melt, aes(y=value, color=Order))+
      aes_string(x=predictor)+
      geom_line()+
      xlab(paste0(predictor, " (transformed polynomials)"))+
      ylab("Transformed value")+
      scale_color_brewer(palette="Set1")+
      theme_bw()
    
    print(poly.plot)
  }
  
  # restore correct column names
  colnames(df)[colnames(df) == "temp.predictor.index"] <- paste0(predictor,".Index")
  return(df)
}

data <-read.csv(file= 'prox_shift_sims.csv')	
data.gca <- code.poly(df=data, predictor="var_val", poly.order=2, orthogonal=TRUE, draw.poly=FALSE)
summ_data<-summaryBy(pk_freq+freq_span+pk_pow+dur~var_val, data=data, keep.names = TRUE)

ggplot(summ_data, aes(x=var_val,y=pk_freq)) +
  geom_line()
model<-lm(pk_freq~poly1+poly2, data=data.gca)
broom::tidy(model, effects = "fixed")
drop1(model, ~., test="Chi")
Anova(model)

ggplot(summ_data, aes(x=var_val,y=freq_span)) +
  geom_line()
model<-lm(freq_span~poly1+poly2, data=data.gca)
broom::tidy(model, effects = "fixed")
drop1(model, ~., test="Chi")
Anova(model)

ggplot(summ_data, aes(x=var_val,y=pk_pow)) +
  geom_line()
model<-lm(pk_pow~poly1+poly2, data=data.gca)
broom::tidy(model, effects = "fixed")
drop1(model, ~., test="Chi")
Anova(model)

ggplot(summ_data, aes(x=var_val,y=dur)) +
  geom_line()
model<-lm(dur~poly1+poly2, data=data.gca)
broom::tidy(model, effects = "fixed")
drop1(model, ~., test="Chi")
Anova(model)



data <-read.csv(file= 'dist_strength_sims.csv')	
data.gca <- code.poly(df=data, predictor="var_val", poly.order=2, orthogonal=TRUE, draw.poly=FALSE)
summ_data<-summaryBy(pk_freq+freq_span+pk_pow+dur~var_val, data=data, keep.names = TRUE)

ggplot(summ_data, aes(x=var_val,y=pk_freq)) +
  geom_line()
model<-lm(pk_freq~poly1+poly2, data=data.gca)
broom::tidy(model, effects = "fixed")
drop1(model, ~., test="Chi")
Anova(model)

ggplot(summ_data, aes(x=var_val,y=freq_span)) +
  geom_line()
model<-lm(freq_span~poly1+poly2, data=data.gca)
broom::tidy(model, effects = "fixed")
drop1(model, ~., test="Chi")
Anova(model)

ggplot(summ_data, aes(x=var_val,y=pk_pow)) +
  geom_line()
model<-lm(pk_pow~poly1+poly2, data=data.gca)
broom::tidy(model, effects = "fixed")
drop1(model, ~., test="Chi")
Anova(model)

ggplot(summ_data, aes(x=var_val,y=dur)) +
  geom_line()
model<-lm(dur~poly1+poly2, data=data.gca)
broom::tidy(model, effects = "fixed")
drop1(model, ~., test="Chi")
Anova(model)


data <-read.csv(file= 'prox_strength_sims.csv')	
data.gca <- code.poly(df=data, predictor="var_val", poly.order=2, orthogonal=TRUE, draw.poly=FALSE)
summ_data<-summaryBy(pk_freq+freq_span+pk_pow+dur~var_val, data=data, keep.names = TRUE)

ggplot(summ_data, aes(x=var_val,y=pk_freq)) +
  geom_line()
model<-lm(pk_freq~poly1+poly2, data=data.gca)
broom::tidy(model, effects = "fixed")
drop1(model, ~., test="Chi")
Anova(model)

ggplot(summ_data, aes(x=var_val,y=freq_span)) +
  geom_line()
model<-lm(freq_span~poly1+poly2, data=data.gca)
broom::tidy(model, effects = "fixed")
drop1(model, ~., test="Chi")
Anova(model)

ggplot(summ_data, aes(x=var_val,y=pk_pow)) +
  geom_line()
model<-lm(pk_pow~poly1+poly2, data=data.gca)
broom::tidy(model, effects = "fixed")
drop1(model, ~., test="Chi")
Anova(model)

ggplot(summ_data, aes(x=var_val,y=dur)) +
  geom_line()
model<-lm(dur~poly1+poly2, data=data.gca)
broom::tidy(model, effects = "fixed")
drop1(model, ~., test="Chi")
Anova(model)


data <-read.csv(file= 'dist_width_sims.csv')	
data.gca <- code.poly(df=data, predictor="var_val", poly.order=2, orthogonal=TRUE, draw.poly=FALSE)
summ_data<-summaryBy(pk_freq+freq_span+pk_pow+dur~var_val, data=data, keep.names = TRUE)

ggplot(summ_data, aes(x=var_val,y=pk_freq)) +
  geom_line()
model<-lm(pk_freq~poly1+poly2, data=data.gca)
broom::tidy(model, effects = "fixed")
drop1(model, ~., test="Chi")
Anova(model)

ggplot(summ_data, aes(x=var_val,y=freq_span)) +
  geom_line()
model<-lm(freq_span~poly1+poly2, data=data.gca)
broom::tidy(model, effects = "fixed")
drop1(model, ~., test="Chi")
Anova(model)

ggplot(summ_data, aes(x=var_val,y=pk_pow)) +
  geom_line()
model<-lm(pk_pow~poly1+poly2, data=data.gca)
broom::tidy(model, effects = "fixed")
drop1(model, ~., test="Chi")
Anova(model)

ggplot(summ_data, aes(x=var_val,y=dur)) +
  geom_line()
model<-lm(dur~poly1+poly2, data=data.gca)
drop1(model, ~., test="Chi")
Anova(model)



data <-read.csv(file= 'prox_width_sims.csv')	
data.gca <- code.poly(df=data, predictor="var_val", poly.order=2, orthogonal=TRUE, draw.poly=FALSE)
summ_data<-summaryBy(pk_freq+freq_span+pk_pow+dur~var_val, data=data, keep.names = TRUE)

ggplot(summ_data, aes(x=var_val,y=pk_freq)) +
  geom_line()
model<-lm(pk_freq~poly1+poly2, data=data.gca)
broom::tidy(model, effects = "fixed")
drop1(model, ~., test="Chi")
Anova(model)

ggplot(summ_data, aes(x=var_val,y=freq_span)) +
  geom_line()
model<-lm(freq_span~poly1+poly2, data=data.gca)
broom::tidy(model, effects = "fixed")
drop1(model, ~., test="Chi")
Anova(model)

ggplot(summ_data, aes(x=var_val,y=pk_pow)) +
  geom_line()
model<-lm(pk_pow~poly1+poly2, data=data.gca)
broom::tidy(model, effects = "fixed")
drop1(model, ~., test="Chi")
Anova(model)

ggplot(summ_data, aes(x=var_val,y=dur)) +
  geom_line()
model<-lm(dur~poly1+poly2, data=data.gca)
drop1(model, ~., test="Chi")
Anova(model)