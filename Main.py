# STEP 1: Import core data engineering and machine learning libraries
import pandas as pd
from sklearn.linear_model import LinearRegression

# STEP 2: Create a simple, readable dataset of student placement records
# Features: CGPA, Coding_Rating (e.g., LeetCode/HackerRank out of 5), Internships_Done
# Target: Package_LPA (Salary in Lakhs Per Annum)
data = {
    'CGPA': [6.5, 7.2, 7.8, 8.5, 9.2, 9.6],
    'Coding_Rating': [2.0, 3.0, 3.5, 4.0, 4.5, 5.0],
    'Internships_Done':,
    'Package_LPA': [3.6, 4.5, 6.0, 8.2, 12.0, 15.5]
}

# Convert this data into a structured table (DataFrame)
df = pd.DataFrame(data)

print("--- Campus Placement Historical Dataset ---")
print(df)
print("\n-------------------------------------------")

# STEP 3: Separate inputs (X) from what we want to predict (y)
X = df[['CGPA', 'Coding_Rating', 'Internships_Done']]
y = df['Package_LPA']

# STEP 4: Create and train the Machine Learning model
model = LinearRegression()
model.fit(X, y)

print("[Success] The model has successfully learned the placement trends!")

# STEP 5: Test the model with a completely new student configuration!
# Let's predict the package for a student with: CGPA = 8.2, Coding Rating = 3.8, Internships = 1
new_student = [[8.2, 3.8, 1]]

predicted_package = model.predict(new_student)

print("\n--- Production Inference Evaluation ---")
print(f"🥇 Predicted Campus Placement Package: {predicted_package[0]:.2f} LPA")
