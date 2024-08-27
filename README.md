# Network Congestion Ratios Prediction for Moroccan Cities

This repository contains the code and resources for predicting network congestion ratios across various Moroccan cities using a BiLSTM model. The project is designed to forecast congestion levels based on historical data, enabling telecom operators to optimize network efficiency and infrastructure.

## Table of Contents

1. Project Overview
2. Data Collection
3. Data Preprocessing
4. Model Architecture
5. Training and Evaluation
6. Model Deployment
7. Installation
8. Usage

## Project Overview

![Untitled Diagram drawio](https://github.com/user-attachments/assets/9d8b4939-a141-42b9-a1ba-803c95b87468) 

This project aims to predict network congestion ratios for Moroccan cities by analyzing historical data on network usage, traffic patterns, and environmental factors. The predictions are intended to help telecom operators make data-driven decisions to enhance network performance and resource allocation.


## Data Collection

The dataset used in this project includes:

- Network traffic data: Historical records of network usage across different cities.
- Environmental data: Information such as temperature, precipitation, wind speed, and holidays, which may impact network congestion.
- News Data: Headlines and summaries related to various categories, such as politics, society, and economy, were collected and analyzed to gauge public sentiment.

## Data Preprocessing

Data preprocessing involved several critical steps:

- Handling Missing Values: Missing data points were filled using interpolation techniques.
- Feature Engineering: Additional features like temperature, rain, wind speed and holiday were included to improve prediction accuracy.
- BERT Embeddings: The news headlines and summaries were processed using BERT embeddings to capture the sentiment and semantic meaning, which were then incorporated into the dataset.
- Data Normalization: Data was normalized to ensure that all features contributed equally to the model.

## Model Architecture

We utilized a Bidirectional Long Short-Term Memory (BiLSTM) model for predicting congestion ratios. The architecture was chosen for its ability to capture temporal dependencies in time series data and to incorporate textual features from news sentiment. Key features of the model include:

- Bidirectional LSTM layers: To capture patterns in both past and future time steps.
- Dense layers: For final predictions.
- Dropout layers: To prevent overfitting and improve generalization.
- BERT Embeddings Integration: BERT embeddings derived from news data were integrated into the model to enrich the feature set.

## Training and Evaluation

The model was trained using the processed dataset:

- Training: The model was trained on 70% of the data, with the remaining 30% used for validation and testing.
- Loss Function: Mean Squared Error (MSE) was used as the loss function.
- Optimization: The model was optimized using the Adam optimizer.
- Evaluation: Performance was evaluated using metrics like Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE).

## Model Deployment

The trained BiLSTM model was deployed through a web-based interface using Flask. The deployment process included:

- Flask GUI: A user-friendly web application was developed to allow users to input data and receive congestion predictions.

  <img width="401" alt="image" src="https://github.com/user-attachments/assets/572bd5bc-2f02-40b0-88ec-3ddf07cd9f45">

- Dockerization: The entire application, including the model and Flask app, was containerized using Docker for easy deployment across different environments.

  <img width="432" alt="image" src="https://github.com/user-attachments/assets/e749fd66-4590-43c9-a488-fa5c462f3e2b">

- Deployment: The Docker image was pushed to Docker Hub for accessibility and ease of deployment.
  
  <img width="441" alt="image" src="https://github.com/user-attachments/assets/b915d482-0d80-4d5e-99a2-70906b063164">

## Installation

To run this project locally, follow these steps:

- Clone the repository:
git clone https://github.com/Nouhailangr/Network-Congestion-Ratios-Prediction/tree/master <br>
cd Network-Congestion-Ratios-Prediction

- Set up Docker (if you wish to use the Dockerized version):<br>
1. Install Docker.<br>
2. Pull the Docker image from Docker Hub:<br>
docker pull nouhailangr/congestion-prediction.1.0<br>
3. Run the Flask application:<br>
python app.py<br>
4. or, using Docker:<br>
docker-compose up

## Usage

After setting up the application, you can:

- Input city and date range: Use the web interface to select a Moroccan city and the forecast period (e.g., next 7 days).
- View predictions: The application will display predicted congestion ratios along with historical data for comparison.
- Analyze trends: Use the visualizations provided to analyze trends and make informed decisions.

  <img width="405" alt="image" src="https://github.com/user-attachments/assets/3b258298-7e9c-43a1-b3b0-48552c06795e">

### Example Output 
  
  <img width="404" alt="image" src="https://github.com/user-attachments/assets/8d1a24f5-8541-43a9-825c-b03c5ac8b7da">

