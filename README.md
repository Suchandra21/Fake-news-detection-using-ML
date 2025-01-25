# Fake-news-detection-using-ML

This project aims to detect fake news using a machine learning approach with a Bi-Directional Long Short-Term Memory (Bi-LSTM) model. The model processes news articles and predicts whether the given news is real or fake.

## Features
- Preprocessing of text data including stemming, stopword removal, and vectorization.
- Random Forest Classifier and vectorization usfin tf-idf
- Confusion matrix, classification report, and accuracy score for performance evaluation.
- K-fold cross-validation for robustness analysis.

## Getting Started
### Prerequisites
- Python 3.8 or above
- Required libraries: `pandas`, `nltk`, `sklearn` , `matplotlib`, `seaborn`, `joblib`

### Installation
1. Clone this repository:
    ```bash
    git clone https://github.com/your-repo/fake-news-detection.git
    cd fake-news-detection
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Usage
1. Prepare the dataset by placing it in the root folder of the project and renaming it to `fake_and_real_news.csv`.
2. Run the main script to train the model:
    ```bash
    python final_fake_news_prediction_using_ML_(1).ipynb
    ```
3. Predict using the model by modifying the input text in the script.

### Dataset
The dataset contains two classes: real news and fake news. It is loaded from a CSV file.

## Model Summary
The ML model uses Random Forest classifier, followed by many NLP processes like stemming, tf-idf, stopwords removal and simple lower casing of the news articles.

## Results
- Accuracy score and confusion matrix are used for evaluation.
- Model achieved an average accuracy of 0.99% (based on K-fold cross-validation).

## Author
Suchandra Das - in collaboration with Atasi Das, Ayush Kumar, Sudip Mahapatra, Ankush Panja, Surojit Biswas, Mohit Singh.
