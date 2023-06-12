---
  title: "Predictive Modeling of Mouse Behavior Using Neural Activity and Visual Stimuli"
output: html_document
---
  
  ## Abstract
  
  This project dives deep into the experimental data collected by Steinmetz et al. (2019) from mice undergoing visual discrimination tasks. By leveraging this intricate dataset, our primary goal was to devise a model capable of accurately predicting trial outcomes based on recorded neural activity and varying contrast visual stimuli. The report provides a comprehensive account of our exploratory data analysis, identification of shared patterns and unique traits across trials, and subsequent data integration. Using these insights, we formulated a logistic regression model incorporating several key variables including the contrast of visual stimuli and the principal components of the neural firing rates across various brain areas. Upon validation on a reserved test set, our model exhibited promising performance, accurately predicting trial outcomes with an approximate accuracy of 70.8%. Despite the encouraging results, we acknowledge potential areas of improvement and contemplate more sophisticated modeling techniques and expanded feature sets for future endeavors in this domain.

## Section 1: Introduction
In the study conducted by Steinmetz et al. (2019), experiments were performed on mice, monitoring their neural activities and responses to visual stimuli. The data was collected over 39 sessions with several hundred trials in each. Our objective in this project is to analyze this data, specifically the spike trains of neurons from the onset of stimuli to 0.4 seconds post-onset, and build a predictive model for the outcome of each trial.

```{r, echo = FALSE, message = FALSE}
# Your R code here...
## Set working directory
setwd("/Users/queena/STA141A-final/sessions")

## Load packages
library("dplyr")

## read in data
getwd()
session=list()
for(i in 1:18){
  session[[i]]=readRDS(paste('./session',i,'.rds',sep=''))
  # Record unique brain_area
  session[[i]]$unique_brain_area <- unique(session[[i]]$brain_area)
  # Record neuron number
  session[[i]]$neuron_number <- dim(session[[i]]$spks[[1]])[1]
  
  # Record number of trials
  
  session[[i]]$trials <- length(session[[i]]$contrast_left)
  # print(session[[i]]$mouse_name)
  # print(session[[i]]$date_exp)
  
}

# Load the first session data for exploration
first_session <- session[[1]]

### Descriptive statistics for each of the component of the dataset
## Number of neurons
neurons_count <- sapply(session, function(x) x$neuron_number)

## Number of trials
trials_count <- sapply(session, function(x) x$trials)


```
## Section 2: Exploratory analysis
In this section, we will present descriptive statistics for key variables across the dataset, followed by more detailed analyses of the first session to provide deeper insights.
#### Neuron Count Across All Sessions
In all sessions, the neuron count ranges from 474 to 1769, with a mean of 905.8. The distribution is skewed towards higher neuron counts, with the 3rd quartile being at 1086.8. The median neuron count across all sessions is 822.5, indicating that at least 50% of the sessions have more than 822 neurons recorded.
```{r, echo=FALSE}
# five number summary
summary(neurons_count)
```

#### Left Contrast Distribution: First Session
Looking at the left contrast distribution in the first session, we see that it ranges from 0 to 1, with a mean of 0.2961. This suggests that on average, the left contrast is closer to the minimum than the maximum. The distribution is heavily skewed towards lower values, as evidenced by the fact that the 1st quartile equals the minimum, and the median (0.25) is close to the 1st quartile.


```{r, echo=FALSE}
summary(first_session$contrast_left)
## Histogram of contrast_left for first_session
hist(first_session$contrast_left, main = "Histogram of contrast_left for first_session", xlab = "contrast_left")
```

#### Right Contrast Distribution: First Session
For the right contrast in the first session, the values also range from 0 to 1 but with a slightly higher mean of 0.4298. This suggests that right contrasts in the session have, on average, slightly higher values than the left contrasts. The median for the right contrast (0.375) is larger than the median for the left contrast, further supporting this observation.
```{r, echo=FALSE}
summary(first_session$contrast_right)
## Histogram of contrast_right for first_session
hist(first_session$contrast_right, main = "Histogram of contrast_right for first_session", xlab = "contrast_right")
```

#### Trial Counts Across All Sessions
The number of trials per session varies between 114 and 447, with an average of 282.3 trials per session. This average is slightly higher than the median (261), suggesting a skew towards higher trial counts in some sessions.
```{r, echo=FALSE}
# five number summary
summary(trials_count)
```

#### Feedback Types: First Session
In the first session, feedback was divided into two categories, labeled "-1" and "1". There were 45 instances of "-1" feedback and 69 instances of "1" feedback. This suggests a slightly more frequent occurrence of the "1" feedback type in the first session.
```{r, echo=FALSE}
## count of feedback_type (-1 vs 1) for first_session
table(first_session$feedback_type)
```

#### Brain Areas: First Session
Eight distinct brain areas were monitored in the first session: ACA, CA3, DG, LS, MOs, root, SUB, and VISp. The number of recorded instances per area ranged from a low of 18 in the root area to a high of 178 in the VISp area.
```{r, echo=FALSE}
## count of brain_area for first_session 
table(first_session$brain_area)
```
Note that each session contains a different combination of brain areas, with some areas being monitored in all sessions and others in only a few. Therefire, we will need to account for this variation in our subsequent analyses.

In summary, our data indicates a good variety in neuron counts, contrasts, trials, feedback types, and brain areas across sessions. The session-to-session variations in these metrics point to a richness of information that could prove valuable in our subsequent analyses.

## Section 3: Data Integration
In order to effectively integrate our data across all sessions and trials, we employ a two-step approach. Firstly, we transform the data to enable effective integration and comparison across sessions, and then we reduce the data's dimensionality using principal component analysis (PCA).

#### Data Transformation
Given the dataset's structure, with each session containing multiple spikes (spks) representing the neuron firing rates, our first step involves transforming the spks into a more digestible form. We calculate the mean firing rate for each neuron across all trials in a session. This transformation simplifies the overall data structure, focusing on the average neuron activity during each session.

However, considering that each neuron is associated with a specific brain area, we extend our transformation to calculate the mean firing rate for each neuron across each brain area within a session. This approach provides a clear picture of neuron activity related to distinct brain areas.

#### Data Integration
With our data transformed, we integrate the individual session data into a single dataframe. This consolidated dataset includes the contrast levels on the left and right, feedback type, and the average neuron firing rate across brain areas from each session.

#### Dimensionality Reduction

Despite the data transformation and integration, the consolidated dataset is still quite complex, with each brain area represented as a separate feature. To manage this complexity and to allow for easier analysis, we employ PCA to reduce the data's dimensionality.

We perform PCA on the brain area data and retain the first three principal components, which typically contain the most variance within the dataset. These three components serve as proxies for the original brain area features but condense the information into fewer variables.

With these steps, we have integrated our data across sessions, balancing the need to maintain critical distinctions in the data with the requirement of creating an analyzable, consolidated dataset.

The resulting integrated dataset can now serve as the basis for subsequent predictive modeling and analysis, paving the way for identifying key patterns and relationships within the neural data.

```{r, echo=FALSE}
### Transformation of the data
## For each spk under a session, we calculate the mean firing rate for each neuron (row)


get_df_target_session <- function(target_session) {

  for (j in 1:length(target_session$spks)) {
    target_session$mean_spks[[j]] <- rowMeans(target_session$spks[[j]])    
  }

  ## Each row of the mean_spks is the mean firing rate of a neuron for the corresponding $brian_area. 
  ## We can then calculate the mean firing rate for each neuron across each brain_area

  unique_brain_area <- unique(target_session$brain_area)
  df_brain_area <- data.frame()
  for (k in 1:length(unique_brain_area)) {
    target_brain_area <- unique_brain_area[k]
    target_brain_area_rows <- which(target_session$brain_area == target_brain_area)
    mean_spk_vector = c()
    for (j in 1:length(target_session$spks)) {
      target_mean_spks <- mean(target_session$spks[[j]][target_brain_area_rows,]) 
      mean_spk_vector <- c(mean_spk_vector, target_mean_spks)   
    }
    df_new_row <- data.frame(brain_area = target_brain_area, as.data.frame(t(mean_spk_vector)))
    df_brain_area <- rbind(df_brain_area, df_new_row)
  }
  # View(df_brain_area)

  ## Now create a target session dataframe with the following columns:
  ## contrast_left, contrast_right, feedback_type, df_brain_area
  df_target_session <- data.frame(contrast_left = target_session$contrast_left, contrast_right = target_session$contrast_right, feedback_type = target_session$feedback_type)

  df_transposed_brain_area <- t(df_brain_area)
  colnames(df_transposed_brain_area) <- df_transposed_brain_area[1,]
  df_transposed_brain_area <- df_transposed_brain_area[-1,]
  df_transposed_brain_area <- apply(df_transposed_brain_area, 2, as.numeric)


  ## Perform a PCA on the df_transposed_brain_area, and keep the first 3 components
  pca <- prcomp(df_transposed_brain_area, scale = TRUE)

  df_transposed_brain_area <- data.frame(pca$x[,1:3])



  # View(df_transposed_brain_area)

  df_target_session <- bind_cols(df_target_session, df_transposed_brain_area)
  # View(df_target_session)
  # names(df_target_session)
  df_target_session
}

df_all_sessions <- data.frame()
for (i in 1:length(session)) {
  target_session <- session[[i]]
  df_target_session <- get_df_target_session(target_session)


  df_all_sessions <- bind_rows(df_all_sessions, df_target_session)
}

head(df_all_sessions)
```

## Section 4: Predictive Modeling
Following the data integration process, we proceed with creating a predictive model. The aim of this model is to leverage the gathered data to predict the feedback type based on the observed contrasts (left and right) and the first three principal components representing average neuron activity across brain areas.

#### Model Preparation
In preparation for model building, we first transform the feedback type into binary form, converting "-1" to "0" and keeping "1" as is. This binary outcome is well-suited for logistic regression, which we will use for our predictive model.

Following this, we divide the dataset into training and testing sets. This split allows us to train our model on one set of data and validate its performance on another. Here, we allocate 80% of the data to the training set and the remaining 20% to the testing set.

#### Model Building
We construct a logistic regression model, using a logit link function suitable for our binary outcome. The predictors in our model include 'contrast_left', 'contrast_right', and the first three principal components (PC1, PC2, PC3). The response variable is the binary 'feedback_type'.

The resulting model is as follows:
  
```{r, echo=FALSE}
## Finally, ready to build a model
## We will use the following columns as predictors: contrast_left, contrast_right, PC1, PC2, PC3
## We will use the following column as the response: feedback_type

## Note that the feedback_type is -1 or 1, so we will transform it into 0 and 1, then use a binomial model with a logit link function
df_all_sessions$feedback_type_transformed <- ifelse(df_all_sessions$feedback_type == -1, 0, 1)


## Split the data into training and testing set
set.seed(123)
train_index <- sample(1:nrow(df_all_sessions), 0.8*nrow(df_all_sessions))
train <- df_all_sessions[train_index,]
test <- df_all_sessions[-train_index,]

## Build a model
model <- glm(feedback_type_transformed ~ contrast_left + contrast_right + PC1 + PC2 + PC3, data = train, family = binomial(link = "logit"))
model$call
```

#### Model Summary
Upon examining the model's summary, we notice several points of interest. The left contrast appears to have a significant positive effect on the feedback type. A unit increase in the left contrast level increases the log-odds of positive feedback by 0.20, holding all other variables constant.

Among the principal components, PC1 exhibits a significant negative effect on the feedback type. This outcome suggests that an increase in the average firing rate of neurons (as captured by PC1) results in a decrease in the log-odds of positive feedback.

However, contrast_right, PC2, and PC3 are not significant at the standard thresholds (p > 0.05), indicating that these variables do not have a substantial effect on the feedback type when all variables are considered.

Overall, our model, with an AIC of 4882, provides us with a useful tool to understand the impact of different variables on the feedback type. The next step is to validate this model on the testing dataset and evaluate its predictive accuracy.

```{r, echo=FALSE}
summary(model)
```

## Section 5: Prediction Performance On The Test Sets
After creating our predictive model, we move on to evaluate its performance. We do this by applying our model to the test set, which is the data that our model hasn't seen during its training phase. This process allows us to assess how well our model generalizes to unseen data and to quantify its predictive accuracy.
#### Prediction on Test Data

With our logistic regression model at hand, we predict the feedback type for our test set. The 'predict' function provides the predicted probabilities of positive feedback for the test data. Following this, we convert these probabilities to binary feedback predictions. We use a threshold of 0.5; probabilities greater than 0.5 are converted to positive feedback (1), and those less than or equal to 0.5 are converted to negative feedback (-1).

```{r, echo=T}
## Predict on the test set
test$predicted <- predict(model, test, type = "response")

## Calculate the accuracy
test$predicted <- ifelse(test$predicted > 0.5, 1, -1)
```
#### Model Accuracy
The final step in our analysis is to calculate the accuracy of our model, defined as the proportion of correct predictions made out of all predictions. We find that our model achieved an accuracy of approximately 70.8% on the test data.

```{r, echo=T}
accuracy <- sum(test$predicted == test$feedback_type)/nrow(test)
print(accuracy)
```
This accuracy is a strong indicator of the model's performance, showing that the model can correctly predict the feedback type in more than 70% of the cases based on the given predictors.

Although this is a promising result, it also suggests room for improvement. Future work may explore different modeling techniques, incorporate more predictors, or fine-tune the model parameters to increase predictive accuracy.

## Section 6: Discussion

Our objective in this project was to build a predictive model capable of forecasting the outcome of each trial based on neural activity data and varying contrast visual stimuli. We started by investigating the Steinmetz et al. (2019) dataset, which included neural responses from mice exposed to a visual discrimination task. We discovered key differences and similarities across trials, identifying a set of shared patterns. By accounting for the differences and incorporating the shared patterns across trials, we were able to integrate the data effectively, setting a solid foundation for further analysis.

The subsequent step involved predictive modeling. Our model of choice was logistic regression, with variables including left contrast, right contrast, and the first three principal components of the neural firing rates across brain areas. The model proved to be effective, as seen from the significant p-values of the coefficients for the left contrast and the first principal component.

Evaluation on the test set revealed that our model was able to predict the outcome of the trials with an accuracy of approximately 70.8%. This result signifies that our model has reasonably good predictive power, demonstrating the potential for predicting trial outcomes based on neural activity and visual stimuli.

However, this study is not without limitations. The assumption of no session effect may not hold in all cases. Additionally, we could look into more complex models and incorporate more predictors to potentially increase the predictive accuracy.

Future work could involve more sophisticated machine learning techniques, including neural networks or random forests, that could potentially model the data with higher accuracy. Additionally, more extensive feature engineering and inclusion of interaction effects might improve the model's performance.