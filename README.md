# Prediction of Parkinson’s Disease Using Voice Analysis


 # Parkinson Detection Using Voice Analysis

## Overview

Parkinson’s Disease (PD) is a progressive neurological disorder caused by the loss of dopamine-producing neurons in the brain. It affects movement, speech, and cognitive functions. Early diagnosis is challenging because symptoms often develop gradually and there is no definitive laboratory test for detection.

This project uses Machine Learning techniques to detect Parkinson’s Disease from voice recordings. Since speech impairments are common in PD patients, voice analysis provides a non-invasive and cost-effective method for early screening.

## Project Objectives

* Develop a machine learning model to detect Parkinson’s Disease using voice features.
* Compare multiple classification algorithms and evaluate their performance.
* Minimize false negative predictions to improve early disease detection.
* Provide a user-friendly web application for prediction using Flask.

## Dataset Information

The dataset was obtained from Kaggle and contains voice measurements from healthy individuals and Parkinson’s patients.

### Dataset Details

* Total Records: 195
* Total Features: 24
* Input Features: 23
* Target Variable: Status

  * 1 = Parkinson’s Disease
  * 0 = Healthy

## Features Used

| Feature Category      | Description                               |
| --------------------- | ----------------------------------------- |
| MDVP:Fo(Hz)           | Average vocal fundamental frequency       |
| MDVP:Fhi(Hz)          | Maximum vocal fundamental frequency       |
| MDVP:Flo(Hz)          | Minimum vocal fundamental frequency       |
| Jitter Features       | Variations in vocal frequency             |
| Shimmer Features      | Variations in amplitude                   |
| NHR, HNR              | Noise-to-harmonics ratio measurements     |
| RPDE, D2              | Nonlinear dynamical complexity measures   |
| DFA                   | Signal fractal scaling exponent           |
| Spread1, Spread2, PPE | Nonlinear measures of frequency variation |

## Technologies Used

* Python
* Flask
* Scikit-Learn
* Pandas
* NumPy
* HTML
* CSS
* SQLite

## Machine Learning Models Evaluated

* Support Vector Machine (SVM)
* K-Nearest Neighbors (KNN)
* Logistic Regression
* Decision Tree
* Random Forest

## Evaluation Metrics

The models were evaluated using:

### Accuracy

Measures the proportion of correctly classified instances.

### Precision

Measures how many predicted positive cases are actually positive.

### Recall

Measures how many actual positive cases are correctly identified.

### AUC Score

Measures the model's ability to distinguish between positive and negative classes.

## Results

After comparing multiple machine learning algorithms:

* SVM achieved the highest performance.
* KNN also produced strong results.
* SVM obtained the highest AUC score and overall classification performance.

Therefore, the Support Vector Machine (SVM) model was selected and saved as a pickle file for deployment.

## Project Structure

```text
Parkinson-Detection-Using-Voice-Analysis/
│
├── app.py
├── model.pkl
├── scaler.pkl
├── data.csv
├── users.db
├── requirements.txt
│
├── templates/
│   ├── index.html
│   ├── login.html
│   ├── register.html
│   └── result.html
│
├── static/
│
└── README.md
```

## Installation

1. Clone the repository

```bash
git clone https://github.com/your-username/Parkinson-Detection-Using-Voice-Analysis.git
```

2. Navigate to the project directory

```bash
cd Parkinson-Detection-Using-Voice-Analysis
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Run the application

```bash
python app.py
```

5. Open your browser and visit

```text
http://127.0.0.1:5000
```

## Features

* User Registration and Login
* Parkinson’s Disease Prediction
* Voice Feature Analysis
* Machine Learning-Based Classification
* Simple and Interactive Web Interface

## Future Enhancements

* Real-time voice recording support
* Deep Learning-based prediction models
* Improved dataset size and diversity
* Cloud deployment
* Doctor recommendation system

## Conclusion

This project demonstrates how machine learning can assist in the early detection of Parkinson’s Disease through voice analysis. By leveraging speech characteristics and classification algorithms, the system provides a fast, non-invasive, and accessible screening solution. Among all evaluated models, SVM achieved the best performance and was selected for deployment.

## Reference

Little MA, McSharry PE, Roberts SJ, Costello DAE, Moroz IM.
"Exploiting Nonlinear Recurrence and Fractal Scaling Properties for Voice Disorder Detection."
BioMedical Engineering OnLine, 2007.

