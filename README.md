# Network Congestion Ratios Prediction for Moroccan Cities

This repository contains the code and resources for predicting network congestion ratios across various Moroccan cities using a BiLSTM model. The project is designed to forecast congestion levels based on historical data, enabling telecom operators to optimize network efficiency and infrastructure.

## Table of Contents

Project Overview
Data Collection
Data Preprocessing
Model Architecture
Training and Evaluation
Model Deployment
Installation
Usage

## Project Overview

This project aims to predict network congestion ratios for Moroccan cities by analyzing historical data on network usage, traffic patterns, and environmental factors. The predictions are intended to help telecom operators make data-driven decisions to enhance network performance and resource allocation.

![Uploading Untitled Diagram.drawio.png…]()

## Data Collection

The dataset used in this project includes:

Network traffic data: Historical records of network usage across different cities.
Environmental data: Information such as temperature, precipitation, wind speed, and holidays, which may impact network congestion.
News Data: Headlines and summaries related to various categories, such as politics, society, and economy, were collected and analyzed to gauge public sentiment.

## Data Preprocessing

Data preprocessing involved several critical steps:

Handling Missing Values: Missing data points were filled using interpolation techniques.
Feature Engineering: Additional features like temperature, rain, wind speed and holiday were included to improve prediction accuracy.
BERT Embeddings: The news headlines and summaries were processed using BERT embeddings to capture the sentiment and semantic meaning, which were then incorporated into the dataset.
Data Normalization: Data was normalized to ensure that all features contributed equally to the model.

## Model Architecture

We utilized a Bidirectional Long Short-Term Memory (BiLSTM) model for predicting congestion ratios. The architecture was chosen for its ability to capture temporal dependencies in time series data and to incorporate textual features from news sentiment. Key features of the model include:

Bidirectional LSTM layers: To capture patterns in both past and future time steps.
Dense layers: For final predictions.
Dropout layers: To prevent overfitting and improve generalization.
BERT Embeddings Integration: BERT embeddings derived from news data were integrated into the model to enrich the feature set.

## Training and Evaluation

The model was trained using the processed dataset:

Training: The model was trained on 70% of the data, with the remaining 30% used for validation and testing.
Loss Function: Mean Squared Error (MSE) was used as the loss function.
Optimization: The model was optimized using the Adam optimizer.
Evaluation: Performance was evaluated using metrics like Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE).

## Model Deployment

The trained BiLSTM model was deployed through a web-based interface using Flask. The deployment process included:

Flask GUI: A user-friendly web application was developed to allow users to input data and receive congestion predictions.
Dockerization: The entire application, including the model and Flask app, was containerized using Docker for easy deployment across different environments.
Deployment: The Docker image was pushed to Docker Hub for accessibility and ease of deployment.

## Installation

To run this project locally, follow these steps:

Clone the repository:
git clone https://github.com/Nouhailangr/Network-Congestion-Ratios-Prediction.git
cd Network-Congestion-Ratios-Prediction

Install dependencies:
pip install -r requirements.txt

Set up Docker (if you wish to use the Dockerized version):
Install Docker.
Pull the Docker image from Docker Hub:
docker pull nouhailangr/congestion-prediction.1.0
Run the Flask application:
python app.py
or, using Docker:
docker-compose up

