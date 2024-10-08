# NLP Document Classification


## Introduction
This project tackles the task of document classification, a core problem in supervised machine learning. Document classification involves assigning categories to text documents, such as news articles, emails, or forum posts. The goal is to demonstrate how machine learning techniques can be effectively applied to classify documents into different categories.

## Objective
The objective is to develop a robust document classification system using the 20 Newsgroups dataset, which consists of approximately 20,000 documents divided into 20 different categories. This system can be used in various applications such as spam filtering, sentiment analysis, or content-based recommendations.

## Approach
The project follows these steps:

- Problem Understanding: Understand the task of document classification and the dataset.
- Data Exploration & Preprocessing: Explore and clean the data by removing metadata, emails, numbers, special characters, and stopwords. The text is tokenized and lemmatized.
- Model Building & Training: Build and train models using Support Vector Machines (SVM) and Naive Bayes, fine-tuning hyperparameters.
- Performance Evaluation: Evaluate model performance using metrics like F1-score and ROC-AUC.
- Model Deployment: Deploy the best model using Flask for real-time document classification.
  
## Features
- Data Preprocessing: Text preprocessing using custom functions and SpaCy.
- Modeling: Models built using Naive Bayes and SVM, trained with TF-IDF features.
- Deployment: A Flask application is used to deploy the best model for document classification.
- Visualization: Word clouds and confusion matrices are generated to visualize insights.
  
## Key Files
- NLP_Classification_Project.ipynb: Jupyter notebook containing the full workflow.
- newsgroup_flask_app.py: Flask application for deployment.
- preprocessing.py: Script for preprocessing the dataset.
- best_fine_tuned_model.pkl: Saved fine-tuned model.
- tfidf_vectorizer.pkl: Saved TF-IDF vectorizer.
- requirements.txt: Dependencies required to run the project.
  
## Installation and Usage
### Prerequisites
- Python 3.7+
- Required libraries listed in requirements.txt
- 
### Installation
- Clone the repository:
git clone https://github.com/kostas696/My_Latest_Projects/edit/main/NLP_Newsgroups_Classification_Deployment.git

- Install dependencies:
pip install -r requirements.txt

- Run the Flask application:
python newsgroup_flask_app.py

- Access the app at http://127.0.0.1:5000/.

## Model Evaluation
The best model, SVM with TF-IDF features extracted from SpaCy-preprocessed text, achieved:
- F1-Score: 0.812
- ROC-AUC Score: 0.984
These metrics indicate robust performance in classifying documents across different categories.

## Visualizations
- Word Clouds: Visualize the most common words in each newsgroup.
- Confusion Matrix: Understand the model’s performance in each category.
- Learning Curve: Assess how the model improves with more training data.
- 
## Conclusion
This project demonstrates how natural language processing and machine learning techniques can be effectively combined to create a powerful document classification system. The deployed model offers real-time predictions and shows high performance across different categories.

## License
This project is licensed under the MIT License
