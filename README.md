# Online-payment-fraud-detection
This is the project our team has done as minor project in our college

Online Payment Fraud Detection using Machine Learning
This project aims to build an intelligent system that detects fraudulent online payment transactions using machine learning techniques. By analyzing transactional patterns and behaviors, the system identifies suspicious activities and flags potential fraud in real time.

📌 Key Features:
Machine Learning Models: Implemented and compared multiple models including Random Forest and Logistic Regression.

Best Model Selection: Random Forest was selected due to its higher accuracy and better performance on evaluation metrics.

Data Preprocessing: Handled imbalanced data using SMOTE, normalized features with MinMaxScaler, and encoded categorical variables using LabelEncoder.

Model Evaluation: Evaluated using metrics such as Accuracy, Precision, Recall, F1 Score, and Confusion Matrix.

Web Interface: Built a user-friendly Flask-based web application to upload transaction data and predict fraud probability.

Deployment: Trained model was serialized using Pickle for real-time inference.

🛠️ Tools & Technologies:
Python, and Flask,
libraries-scikit_learn, Pandas, NumPy,SMOTE for oversampling,
HTML/CSS for web interface

Random Forest Accuracy: ~99%

Effective detection of minority class (fraud cases) with good precision and recall
