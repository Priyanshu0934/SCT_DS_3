📘 Decision Tree Classifier on Bank Marketing Dataset
🔍 Project Objective
The goal of this project is to use a Decision Tree Classifier to predict whether a customer will subscribe to a term deposit (deposit = yes or no) based on various personal, financial, and marketing attributes provided by a bank.

🧠 What is a Decision Tree?
A Decision Tree is a supervised machine learning algorithm used for classification and regression. It works by splitting the dataset into smaller subsets based on feature conditions, forming a tree-like structure where each internal node is a decision, and each leaf node is a prediction.

📊 Dataset Description
The dataset includes features such as:
age, job, marital, education
Financial details: balance, housing, loan
Marketing info: contact, campaign, duration
Target variable: deposit (whether the client subscribed)

⚙️ Steps Involved
 Data Loading – Reading the Excel file using pandas.
 Preprocessing – Encoding categorical variables using LabelEncoder.
 Splitting Data – Using train_test_split (70% train, 30% test).
 Model Training – Using DecisionTreeClassifier from sklearn.
 Evaluation – Accuracy, classification report, and confusion matrix.
 Visualization – Tree plotted using plot_tree() for interpretability.

✅ Results
Achieved ~78% accuracy on test data.
The model was able to classify subscription behavior with reasonable performance.
Decision tree visualization was generated with max_depth=4 for better interpretability.

📦 Technologies Used
Python
Pandas, NumPy
Scikit-learn
Matplotlib
