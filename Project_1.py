import pandas as pd

# Load the CSV file
df = pd.read_csv('Student Performance Predictor for EduQuest Coaching.csv')

# Display the first 5 rows
print(df.head().to_markdown(index=False, numalign="left", stralign="left"))

# Display the column names and their data types
print(df.info())

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Identify categorical columns
categorical_cols = df.select_dtypes(include='object').columns
print(f"Categorical columns: {categorical_cols.tolist()}")

# Apply one-hot encoding
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Separate features (X) and target (y)
X = df_encoded.drop('final_exam_score', axis=1)
y = df_encoded['final_exam_score']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nShape of X_train: {X_train.shape}")
print(f"Shape of X_test: {X_test.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of y_test: {y_test.shape}")

# Initialize and train the RandomForestRegressor model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R-squared (R2) Score: {r2:.2f}")

import altair as alt

# Create a DataFrame for plotting
results_df = pd.DataFrame({'Actual Final Exam Score': y_test, 'Predicted Final Exam Score': y_pred})

# Create the scatter plot
chart = alt.Chart(results_df).mark_point().encode(
    x=alt.X('Actual Final Exam Score', axis=alt.Axis(title='Actual Final Exam Score')),
    y=alt.Y('Predicted Final Exam Score', axis=alt.Axis(title='Predicted Final Exam Score')),
    tooltip=['Actual Final Exam Score', 'Predicted Final Exam Score']
).properties(
    title='Actual vs. Predicted Final Exam Scores'
).interactive() # Make the chart interactive for zooming and panning

# Save the chart as a JSON file
chart.save('actual_vs_predicted_final_exam_scores.json')

