library(dplyr)
library(caret)
library(psych)
library(pROC)

data1 <- read.csv("C:/Users/Kiranmayie/Documents/Fundamentals of data science/week6/heart_disease_health_indicators_BRFSS2015.csv" )

data <- data1 %>% sample_n(1000)

# Remove duplicates if you haven't already
data_no_duplicates <- data %>% distinct()

# Select a subset of relevant variables for the model (optional)
data_model <- data_no_duplicates %>% 
  select(HeartDiseaseorAttack, HighBP, HighChol, BMI, Smoker, Stroke, Diabetes, PhysActivity)

# Ensure that the target variable is a factor
data_model$HeartDiseaseorAttack <- as.factor(data_model$HeartDiseaseorAttack)

set.seed(123)  # For reproducibility
train_index <- createDataPartition(data_model$HeartDiseaseorAttack, p = 0.7, list = FALSE)
train_data <- data_model[train_index, ]
test_data <- data_model[-train_index, ]

logistic_model <- glm(HeartDiseaseorAttack ~ ., data = train_data, family = binomial)
summary(logistic_model)

pred_prob <- predict(logistic_model, newdata = test_data, type = "response")
pred_class <- ifelse(pred_prob > 0.5, 1, 0)

confusion <- confusionMatrix(as.factor(pred_class), test_data$HeartDiseaseorAttack)
print(confusion)

roc_curve <- roc(test_data$HeartDiseaseorAttack, pred_prob)
plot(roc_curve, col = "blue", main = "ROC Curve")
auc_value <- auc(roc_curve)
print(auc_value)

# Create a confusion matrix first
confusion <- confusionMatrix(as.factor(pred_class), test_data$HeartDiseaseorAttack)

# Calculate the accuracy
accuracy <- confusion$overall['Accuracy']
print(accuracy)

accuracy <- mean(pred_class == test_data$HeartDiseaseorAttack)
print(accuracy)

library(MLmetrics)
Precision(as.factor(predicted_class), as.factor(test_data$HeartDiseaseorAttack))
Recall(as.factor(predicted_class), as.factor(test_data$HeartDiseaseorAttack))
F1_Score(as.factor(predicted_class), as.factor(test_data$HeartDiseaseorAttack))

# and 'HeartDiseaseorAttack' column has heart disease info (0 or 1)
data$Sex <- factor(data$Sex, levels = c(0, 1), labels = c("Female", "Male"))

# Create a bar plot to show the distribution of heart disease by sex
ggplot(data, aes(x = Sex, fill = factor(HeartDiseaseorAttack))) +
  geom_bar(position = "dodge") +
  labs(title = "Heart Disease Distribution by Sex",
       x = "Sex",
       y = "Count",
       fill = "Heart Disease or Attack") +
  scale_fill_manual(values = c("0" = "lightblue", "1" = "red")) +
  theme_minimal()


# Visualization for Heart Disease based on High Cholesterol
ggplot(data, aes(x = factor(HighChol), fill = factor(HeartDiseaseorAttack))) +
  geom_bar(position = "dodge") +
  labs(title = "Heart Disease Distribution Based on High Cholesterol",
       x = "High Cholesterol",
       y = "Count",
       fill = "Heart Disease or Attack") +
  scale_fill_manual(values = c("lightgreen", "red")) +
  theme_minimal()

# Visualization for Heart Disease based on Diabetes
ggplot(data, aes(x = factor(Diabetes), fill = factor(HeartDiseaseorAttack))) +
  geom_bar(position = "dodge") +
  labs(title = "Heart Disease Distribution Based on Diabetes",
       x = "Diabetes",
       y = "Count",
       fill = "Heart Disease or Attack") +
  scale_fill_manual(values = c("lightblue", "red")) +
  theme_minimal()
