# Load necessary libraries
library(ggplot2)
library(dplyr)
library(corrplot)
library(ggcorrplot)
library(caTools)
library(reshape2)
library(rpart)
library(caret)
library(pROC)

# Load the dataset
credit_data <- read.csv("C:/Users/Kiranmayie/Documents/Fundamentals of data science/week7/credit_risk_dataset.csv")

# View the first few rows of the dataset
head(credit_data)

# Summary of the dataset
summary(credit_data)

# Summary statistics for continuous variables
summary(select(credit_data, person_age, person_income, person_emp_length, loan_amnt, loan_int_rate))

# Data Preprocessing

# Check for missing values
missing_values <- sapply(credit_data, function(x) sum(is.na(x)))
print(missing_values)

# Encode categorical variables using one-hot encoding
credit_data$loan_intent <- as.factor(credit_data$loan_intent)
credit_data$person_home_ownership <- as.factor(credit_data$person_home_ownership)
credit_data$loan_status <- as.factor(credit_data$loan_status)

#remove duplicate rows
credit_data <- credit_data %>% distinct()

# Descriptive Statistics
# Summary statistics for continuous variables
summary_stats <- credit_data %>%
  summarise(
    mean_loan_amnt = mean(loan_amnt, na.rm = TRUE),
    sd_loan_amnt = sd(loan_amnt, na.rm = TRUE),
    median_loan_amnt = median(loan_amnt, na.rm = TRUE),
    mean_loan_int_rate = mean(loan_int_rate, na.rm = TRUE),
    sd_loan_int_rate = sd(loan_int_rate, na.rm = TRUE),
    median_loan_int_rate = median(loan_int_rate, na.rm = TRUE)
  )

print(summary_stats)

# Visualize the correlation matrix
corrplot(correlation_matrix, method = "circle")

# Scatter plot to visualize the correlation between loan_amnt and person_income
ggplot(credit_data, aes(x = person_income, y = loan_amnt)) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "lm", col = "blue") +
  labs(title = "Correlation between Person Income and Loan Amount",
       x = "Person Income",
       y = "Loan Amount")

# Assuming df is your dataframe
# Visualize categorical columns and numerical columns
categorical_columns <- c("person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file")

# Loop through each categorical column and create a bar plot
for (col in categorical_columns) {
  p <- ggplot(credit_data, aes_string(x = col)) + 
    geom_bar(fill = "steelblue") + 
    theme(axis.text.x = element_text(angle = ifelse(col == "loan_intent", 45, 0), hjust = 1)) +
    labs(title = paste("Distribution of", col), x = col, y = "Count") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5))
  
  ggsave(filename = paste0(col, "_plot.png"), plot = p) # Save each plot as a PNG file
}

numerical_columns <- c("person_age", "person_income", "loan_amnt", "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length")

# Loop through each numerical column and create a histogram
for (col in numerical_columns) {
  h <- ggplot(credit_data, aes_string(x = col)) + 
    geom_histogram(bins = 20, fill = "steelblue", color = "black") +
    labs(title = paste("Distribution of", col), x = col, y = "Frequency") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5)) 
  
  ggsave(filename = paste0(col, "_histogram.png"), plot = h) # Save each plot as a PNG file
}

# Compute the correlation matrix
cor_matrix <- cor(credit_data[numerical_columns], use = "complete.obs")

ggcorrplot(cor_matrix, method = "circle", type = "lower", lab = TRUE, 
           lab_size = 3, colors = c("blue", "white", "red"), 
           title = "Correlation Matrix", ggtheme = theme_minimal())


set.seed(123)

split <- sample.split(credit_data$loan_status, SplitRatio = 0.7)
training_set <- subset(credit_data, split == TRUE)
testing_set <- subset(credit_data, split == FALSE)

logistic_model <- glm(loan_status ~ ., data = training_set, family = binomial)
summary(logistic_model)

# Assuming you have already trained your logistic regression model
# and have predictions and true labels
predicted_probabilities <- predict(logistic_model, newdata = testing_set, type = "response")
predicted_classes <- ifelse(predicted_probabilities > 0.5, 1, 0)
confusion_matrix <- table(testing_set$loan_status, predicted_classes)

print(confusion_matrix)

# Convert the confusion matrix into a long format
cm_melted <- melt(confusion_matrix)

# Plot the confusion matrix
ggplot(cm_melted, aes(Var2, Var1, fill = value)) +
  geom_tile() +
  geom_text(aes(label = value), color = "white") +
  scale_fill_gradient(low = "lightblue", high = "steelblue") +
  labs(x = "Predicted", y = "Actual", title = "Confusion Matrix") +
  theme_minimal()

# Accuracy
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)

# Precision
precision <- confusion_matrix[2, 2] / sum(confusion_matrix[2, ])

# Recall (Sensitivity)
recall <- confusion_matrix[2, 2] / sum(confusion_matrix[, 2])

# Specificity
specificity <- confusion_matrix[1, 1] / sum(confusion_matrix[, 1])

# Display the metrics
cat("Logistic Regression Metrics:\n",
"Accuracy:", accuracy, "\n",
"Precision:", precision, "\n",
"Recall:", recall, "\n",
"Specificity:", specificity, "\n")

# Create an ROC object
roc_obj <- roc(testing_set$loan_status, predicted_probabilities)

# Plot the ROC curve
plot(roc_obj, main = "ROC Curve for Logistic Regression")

# Add AUC to the plot
auc_value <- auc(roc_obj)
legend("bottomright", legend = paste("AUC =", round(auc_value, 2)), col = "blue", lwd = 2)


# Assuming 'credit_data' is your dataset and it has been split into training and testing sets
model <- rpart(loan_status ~ ., data = training_set, method = "class")

# Predict on the test set
predictions <- predict(model, testing_set, type = "class")

# Generate the confusion matrix
confusion_matrix_dt <- confusionMatrix(predictions, testing_set$loan_status)

# Print the confusion matrix
print(confusion_matrix_dt)

# Accuracy
accuracy_dt <- confusion_matrix_dt$overall['Accuracy']

# Precision
precision_dt <- confusion_matrix_dt$byClass['Pos Pred Value']  # 'Pos Pred Value' corresponds to Precision

# Recall
recall_dt <- confusion_matrix_dt$byClass['Sensitivity']

# Specificity
specificity_dt <- confusion_matrix_dt$byClass['Specificity']

# Display the metrics
cat("Decision Tree Metrics:\n",
    "Accuracy:", accuracy_dt, "\n",
  "Precision:", precision_dt, "\n",
  "Recall:", recall_dt, "\n",
  "Specificity:", specificity_dt, "\n")




