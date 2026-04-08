# Housing Prices Prediction (End-to-End ML Project)

## Overview
This project predicts housing prices using machine learning techniques.  
It demonstrates a complete end-to-end ML pipeline including data ingestion, preprocessing, model training, evaluation, and deployment-ready setup.

The project also integrates **MLOps practices** using DVC and Docker to ensure reproducibility and scalability.

---

## Features
- Data preprocessing and feature engineering
- Model training and evaluation
- Experiment tracking with DVC
- Reproducible ML pipeline
- Dockerized environment for deployment
- Config-driven pipeline using YAML files

---

## Tech Stack
- **Python**
- **Pandas, NumPy**
- **Scikit-learn**
- **DVC (Data Version Control)**
- **Docker & Docker Compose**
- **YAML (for pipeline configuration)**

---

## Project Structure
housing-prices-project/
│
├── data/ # Raw and processed data (tracked with DVC)
├── src/ # Source code (data processing, training, etc.)
├── config/ # Configuration files
├── metrics/ # Model evaluation metrics
├── dvc.yaml # DVC pipeline definition
├── params.yaml # Parameters for training
├── Dockerfile # Docker configuration
├── docker-compose.yml # Multi-container setup
├── requirements.txt # Dependencies
└── README.md
---

## ⚙️ Pipeline Workflow
1. Data Ingestion  
2. Data Preprocessing  
3. Model Training  
4. Model Evaluation  
5. Metrics Tracking with DVC  

---

## Results
- Model Performance: *(Add your accuracy / RMSE here)*
- Metrics tracked using DVC for reproducibility

---

## Run with Docker
```bash
docker-compose up --build
