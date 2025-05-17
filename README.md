# My Latest Projects

This repository contains a collection of my latest data science and machine learning projects. Each project highlights specific techniques, tools, and technologies used to solve real-world problems and derive actionable insights.

---

## Table of Contents
1. [Customer Segmentation and Market Basket Analysis for E-commerce Retail](#customer-segmentation-and-market-basket-analysis-for-e-commerce-retail)
2. [Market Price Prediction](#market-price-prediction)
3. [Movie Genre Classification](#movie-genre-classification)
4. [Predictive Modeling for Disease Diagnosis](#predictive-modeling-for-disease-diagnosis)
5. [Credit Card Transactions Fraud Detection](#credit-card-transactions-fraud-detection)
6. [NLP Newsgroups Classification and Deployment](#nlp-newsgroups-classification-and-deployment)
7. [Cab Industry Analysis: Data Exploration, Hypothesis Testing, and Strategic Recommendations](#cab-industry-analysis-data-exploration-hypothesis-testing-and-strategic-recommendations)
8. [New York Housing Market Analysis and Price Prediction](#new-york-housing-market-analysis-and-price-prediction)
9. [Gym Members Calories Prediction with CatBoost](#gym-members-calories-prediction-with-catboost)
10. [Air Quality Prediction](#air-quality-prediction-and-deployment)
11. [Mortgage Propensity Assessment](#mortgage-propensity-assessment)
12. [Crypto Streaming Pipeline: Real-Time Crypto Price Dashboard](#crypto-streaming-pipeline-real-time-crypto-price-dashboard)
13. [AI-Powered Log Analysis – GCP vs. Local LLM (Case Study)](#ai-powered-log-analysis--gcp-vs-local-llm-case-study)

---

## Projects

### Customer Segmentation and Market Basket Analysis for E-commerce Retail
- **Description**: Led a data-driven project focused on customer segmentation, sales trend analysis, and market basket analysis using a dataset from a UK-based online retailer. The project involved in-depth exploration of customer purchasing patterns, segmentation based on Recency, Frequency, Monetary (RFM) analysis, and discovery of product associations using the Apriori algorithm. The outcomes provided valuable insights for enhancing marketing strategies, product placement, and inventory management.
- **Technologies Used**: Python, pandas, Seaborn, Scikit-learn, NetworkX, mlxtend
- **Techniques**: RFM-T Segmentation, Market Basket Analysis, K-Means Clustering, Apriori Algorithm
- **Key Impact**:
  - Identified customer segments for targeted marketing and retention strategies.
  - Discovered high-confidence product association rules for effective cross-selling and product bundling.
  - Provided actionable insights for marketing campaigns, product placement, and inventory management strategies.

---

### Market Price Prediction
- **Description**: Developed a robust time series forecasting model for market analysis, focusing on predicting the quantity and prices of commodities based on historical data. The project involved data preprocessing, exploratory data analysis, feature engineering, model selection, training, and evaluation. Several models were tested, including ARIMA, SARIMA, Prophet, and LSTM, with LSTM models showing significant promise, especially in price forecasting.
- **Technologies Used**: Python, Pandas, NumPy, ARIMA, SARIMA, Prophet, LSTM
- **Key Impact**:
  - Achieved high accuracy in forecasting commodity prices using the LSTM model.
  - Contributed to optimizing inventory management and pricing strategies.
  - Provided actionable insights for market analysis.

---

### Movie Genre Classification
- **Description**: Developed a comprehensive machine learning pipeline to classify movie genres based on descriptions using models such as Logistic Regression, SVM, Random Forest, and XGBoost. Explored feature extraction techniques, including TF-IDF, Word2Vec, and GloVe embeddings. The approach involved preprocessing, model training, evaluation, and deployment.
- **Technologies Used**: Python, Pandas, NumPy, Scikit-learn, XGBoost, Word2Vec, GloVe
- **Key Impact**:
  - Achieved an accuracy of 0.58 with the SVM model using TF-IDF features.
  - Demonstrated significant insights into NLP techniques for text classification.
  - Provided a foundation for recommendation systems.

---

### Predictive Modeling for Disease Diagnosis
- **Description**: Built predictive models to classify individuals into diseased or non-diseased categories based on health attributes. The project aimed to assist healthcare professionals in early detection and personalized patient care.
- **Technologies Used**: Python, Pandas, Scikit-learn, XGBoost, SHAP
- **Key Impact**:
  - Achieved 99.5% accuracy with the XGBoost model.
  - Provided a reliable tool for early disease detection, enhancing patient outcomes.

---

### Credit Card Transactions Fraud Detection
- **Description**: Developed machine learning models to detect fraudulent credit card transactions. The project involved data preprocessing, feature engineering, and extensive exploratory data analysis (EDA).
- **Technologies Used**: Python, Scikit-learn, XGBoost, RandomForest, SMOTE
- **Key Impact**:
  - Built a well-balanced fraud detection system with RandomForest and XGBoost models.
  - Improved precision and recall for fraud detection.

---

### NLP Newsgroups Classification and Deployment
- **Description**: Developed a robust document classification system using the 20 Newsgroups dataset. The system classifies documents into categories, with applications in spam filtering and sentiment analysis.
- **Technologies Used**: Python, Scikit-learn, SpaCy, NLTK
- **Key Impact**:
  - Achieved an F1-score of 0.83 and ROC-AUC score of 0.987.
  - Successfully deployed the model for real-time classification.

---

### Cab Industry Analysis: Data Exploration, Hypothesis Testing, and Strategic Recommendations
- **Description**: Analyzed U.S. cab industry data to identify the most suitable company for investment. The project focused on customer usage patterns, market dynamics, and profitability trends.
- **Technologies Used**: Python, Pandas, Statsmodels
- **Key Impact**:
  - Provided strategic recommendations for investment based on market dynamics.

---

### New York Housing Market Analysis and Price Prediction
- **Description**: Developed a machine learning pipeline for predicting housing prices in New York. Included data collection, exploratory data analysis, model training, and deployment.
- **Technologies Used**: Python, XGBoost, Flask
- **Key Impact**:
  - Achieved a high R^2 score of 0.775 for housing price predictions.
  - Delivered a functional web app for real-time price prediction.

---

### Gym Members Calories Prediction with CatBoost
- **Description**: This project predicts the number of calories burned by gym members during exercise sessions based on health and activity features. The model was trained using CatBoost, achieving high accuracy. It was deployed as a web service via FastAPI, containerized with Docker for seamless deployment. The project emphasizes real-time predictions for personalized fitness planning and progress tracking.

- **Technologies Used**:
Programming Languages: Python
Libraries and Frameworks: CatBoost, FastAPI, SHAP, Pandas, NumPy
Deployment Tools: Docker, Uvicorn
Data Handling: RFE, Feature Engineering, Data Preprocessing
Model Training: CatBoost with hyperparameter tuning (Optuna)

- **Key Impact**:
Achieved a low RMSE of 8.13, indicating high prediction accuracy.
Deployed a scalable web service for real-time calorie predictions.
Enhanced personalized fitness tracking and provided actionable insights for gym members.

---

### Air Quality Prediction and Deployment
- **Description**: Developed and deployed a machine learning-based system to predict air quality levels using a dataset of environmental and demographic metrics. The project included extensive data preprocessing, exploratory data analysis, model selection, and hyperparameter tuning. The final solution was deployed as a web service using FastAPI, Docker, and Kubernetes, with integrated monitoring via Prometheus and Grafana. The deployed application provides real-time air quality predictions, enabling actionable insights for governments, industries, and individuals to mitigate the effects of air pollution.

- **Technologies Used**: Python, pandas, Seaborn, Scikit-learn, CatBoost, XGBoost, LightGBM, FastAPI, Docker, Kubernetes, Prometheus, Grafana, Render

- **Techniques**: Class Imbalance Handling, Weighted Metrics (Weighted F1-Score), Feature Engineering, Optuna Hyperparameter Tuning, Containerization, Cloud Deployment, Monitoring

- **Key Impact**:
Achieved a high Weighted F1-Score of 0.9578 using the CatBoost model, demonstrating its effectiveness in handling imbalanced datasets and predicting critical air quality levels.
Identified key environmental factors like Carbon Monoxide (CO) and proximity to industrial areas as major contributors to poor air quality.
Successfully deployed the application in a production environment, offering an interactive API for real-time air quality predictions.
Integrated monitoring tools (Prometheus and Grafana) for tracking service performance and usage metrics, ensuring reliability and transparency.
Provided actionable insights to stakeholders for improving public health and environmental policies.

---

### Mortgage Propensity Assessment
- **Description**: Built a predictive pipeline to identify high-propensity mortgage customers using labeled retail banking data. The project addressed significant class imbalance (only 1.3% positive class), engineered domain-specific features (e.g., years at current address/job), handled complex date parsing and placeholder values (e.g., 9999-10-01), and applied isotonic calibration with threshold tuning for optimal F1 performance. Inference was performed on a new set of prospects to guide CRM targeting.

- **Technologies Used**: Python, Pandas, NumPy, Scikit-learn, CatBoost, Optuna, SHAP, CalibratedClassifierCV, Matplotlib

- **Key Impact**:
Used threshold calibration (0.103) to significantly improve model decision-making under extreme class imbalance.
Final calibrated model achieved:
F1 Score: 0.203,
Precision: 0.152,
Recall: 0.304.
Identified 56 high-confidence mortgage prospects from a pool of 2,747 new potential customers.
Delivered a data-driven lead scoring file (potential_df_scored.csv) for CRM teams to prioritize outreach.

---

### Crypto Streaming Pipeline: Real-Time Crypto Price Dashboard
- **Description**: Designed and deployed a real-time data streaming pipeline using Google Cloud Platform (GCP) to process live cryptocurrency prices from the OKX WebSocket API. The pipeline includes ingestion via Dockerized Kafka producers, storage in Google Cloud Storage (GCS), transformation using Apache Spark, warehousing in BigQuery, and dynamic dashboard visualization in Looker Studio. The solution is orchestrated using Airflow running on a GCE VM, and infrastructure is provisioned with Terraform.

- **Technologies Used**: Python, Apache Kafka, Apache Spark, Airflow, Google Cloud Platform (GCS, BigQuery, GCE), Looker Studio, Docker, Terraform

- **Key Impact**:
Enabled real-time collection and processing of crypto market data using a scalable, fault-tolerant architecture.
Deployed a dynamic Looker Studio dashboard to visualize pricing trends and volume insights per crypto asset.
Automated infrastructure provisioning and data pipeline execution using Terraform and Airflow, ensuring reproducibility and maintainability.

---

### AI-Powered Log Analysis – GCP vs. Local LLM (Case Study)
- **Description**: Conducted a comparative case study of two approaches to intelligent log analysis and AI-powered root cause resolution. The first solution leverages Google Cloud’s serverless architecture with Vertex AI and Cloud Run for real-time triage. The second solution runs fully locally using a Dockerized EFK stack (Elasticsearch, Filebeat, Kibana) and a local Ollama instance of the LLaMA 3.2 model. Both setups parse ERROR logs and invoke LLMs to generate human-readable explanations and fixes, offering scalable and offline alternatives.

- **Technologies Used**: Vertex AI (Gemini 2.0), Cloud Run, Pub/Sub, Google Cloud Logging, Flask, Ollama, LLaMA 3.2, Docker, Elasticsearch, Filebeat, Kibana, Python

- **Key Impact**:
Demonstrated real-time AI log triage pipeline on GCP using Vertex AI and Cloud-native triggers.
Built an open-source, local alternative using the EFK stack and Ollama for offline inference.
Delivered a comprehensive feature comparison, identified performance and scalability trade-offs, and proposed enhancements including agent-based remediation and RAG pipelines for logs.
