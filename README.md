# Placement Navigator
The Placement Navigator project is designed to streamline and enhance the student placement process by leveraging machine learning and data analysis. It aims to predict a student's likelihood of securing placement and recommend suitable companies based on their skills and academic performance.

Objectives  
1. Predictive Analysis: Develop a robust machine learning model that accurately forecasts a student's placement status based on academic records, chosen stream, and skill set.
2. Company Recommendations: Provide students with a curated list of companies that align with their qualifications and skills, enhancing their chances of successful employment. 

Tools and Technology Requirements 
 1. Programming Languages: Python
 2. Libraries: Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn
 3. Frameworks: Flask for web development
 4. Frontend Technologies: HTML, CSS, JavaScript
 5. IDE: Jupyter Notebook, VS Code 

Decision Tree 
1. Test Accuracy: 	82.6% :Highest, but may overfit 
2. Understands Skills as Features: 	Limited 
3. Performs Well on New Data: 	Weak  tends to memorize training data 
4. Handling Imbalanced Data: 	Needs manual class balancing 
5. Overfitting Tendency: 	High 
6. Ease of Understanding: 	Very easy to interpret 
7. Training Speed: 	Fastest 
8. Suitability for Real Deployment: 	Not suitable tends to  overfit and may perform inconsistently on new data 
9. Final Verdict: 	Rejected due to overfitting 

Random Forest 
1. Test Accuracy: 	78.2% :Balanced performance 
2. Understands Skills as Features: 	Moderate 
3. Performs Well on New Data: 	Better than Decision Tree  
4. Handling Imbalanced Data: 	Handles better with class weight 
5. Overfitting Tendency: 	Moderate
6. Ease of Understanding: 	Moderate complexity  
7. Training Speed: Moderate
8. Suitability for Real Deployment: Suitable for moderate use performs well but may need tuning for larger or more complex datasets 
9. Final Verdict: 	Considered as a secondary option balanced but not the best performer 

XGBoost (Final Model) 
1. Test Accuracy: 	73.9% :Slightly lower, but more reliable on new data 
2. Understands Skills as Features: 	Strongly prioritizes skills like Python, ML, SQL 
3. Performs Well on New Data: 	Excellent generalization on unseen profiles   
4. Handling Imbalanced Data: 	Naturally handles with boosting  
5. Overfitting Tendency: 	Low due to regularization 
6. Ease of Understanding: 	Harder to interpret, but has feature importance charts   
7. Training Speed: Slower, but worth it 
8. Suitability for Real Deployment: Highly suitable  consistently accurate, handles complexity well, and performs reliably in real world scenarios 
9. Final Verdict: 	Chosen Model  offers the best balance of accuracy, generalization, and effectively uses skills as a key feature for prediction 

Future Enhancements: 
1.	Skill Gap Analysis and Personalized Recommendations  
2.	Interactive Chatbot for Career Guidance
3. Job Specific Resume generator 
