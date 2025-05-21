# Placement Navigator
The Placement Navigator project is designed to streamline and enhance the student placement process by leveraging machine learning and data analysis. It aims to predict a student's likelihood of securing placement and recommend suitable companies based on their skills and academic performance.

Objectives  
•	Predictive Analysis: Develop a robust machine learning model that accurately forecasts a student's placement status based on academic records, chosen stream, and skill set. 
•	Company Recommendations: Provide students with a curated list of companies that align with their qualifications and skills, enhancing their chances of successful employment. 

Tools and Technology Requirements 
•	Programming Languages: Python 
•	Libraries: Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn 
•	Frameworks: Flask for web development 
•	Frontend Technologies: HTML, CSS, JavaScript 
•	IDE: Jupyter Notebook, VS Code 

Decision Tree 
Test Accuracy: 	82.6% :Highest, but may overfit 
Understands Skills as Features: 	Limited 
Performs Well on New Data: 	Weak  tends to memorize training data 
Handling Imbalanced Data: 	Needs manual class balancing 
Overfitting Tendency: 	High 
Ease of Understanding: 	Very easy to interpret 
Training Speed: 	Fastest 
Suitability for Real Deployment: 	Not suitable tends to  overfit and may perform inconsistently on new data 
Final Verdict: 	Rejected due to overfitting 

Random Forest 
Test Accuracy: 	78.2% :Balanced performance 
Understands Skills as Features: 	Moderate 
Performs Well on New Data: 	Better than Decision Tree  
Handling Imbalanced Data: 	Handles better with class weight 
Overfitting Tendency: 	Moderate
Ease of Understanding: 	Moderate complexity  
Training Speed: Moderate
Suitability for Real Deployment: Suitable for moderate use performs well but may need tuning for larger or more complex datasets 
Final Verdict: 	Considered as a secondary option balanced but not the best performer 

XGBoost (Final Model) 
Test Accuracy: 	73.9% :Slightly lower, but more reliable on new data 
Understands Skills as Features: 	Strongly prioritizes skills like Python, ML, SQL 
Performs Well on New Data: 	Excellent generalization on unseen profiles   
Handling Imbalanced Data: 	Naturally handles with boosting  
Overfitting Tendency: 	Low due to regularization 
Ease of Understanding: 	Harder to interpret, but has feature importance charts   
Training Speed: Slower, but worth it 
Suitability for Real Deployment: Highly suitable  consistently accurate, handles complexity well, and performs reliably in real world scenarios 
Final Verdict: 	Chosen Model  offers the best balance of accuracy, generalization, and effectively uses skills as a key feature for prediction 

Future Enhancements: 
1.	Skill Gap Analysis and Personalized Recommendations  
2.	Interactive Chatbot for Career Guidance
3. Job Specific Resume generator 
