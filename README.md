# Pump Fault Detection using Machine Learning

This project implements a machine learning based condition monitoring system for an industrial pump using sensor data.

## Problem Statement
Detect pump operating condition and estimate fault probability using pressure, flow, and motor signals.

## Dataset
- Pressure sensors (PS1–PS6)
- Flow sensors (FS1–FS2)
- Motor electrical signal (EPS1)
- Profile labels for pump condition

## Methodology
- Feature extraction using cycle-wise mean values
- Data normalization using StandardScaler
- Model training:
  - Logistic Regression
  - Decision Tree
  - Ensemble Voting Classifier
- Performance evaluation using accuracy, confusion matrix, and classification report

## Output
- Pump condition classification
- Fault probability score
- Automatically generated pump health certificate

## Tools & Libraries
Python, Pandas, Scikit-learn, Matplotlib, Seaborn

## Status
Academic project (1st semester Mechanical Engineering)
