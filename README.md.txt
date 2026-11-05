# SMS Spam Classifier

An NLP and Machine Learning project that automatically classifies SMS messages as **Spam** or **Ham (Legitimate Message)** using Natural Language Processing (NLP) and supervised machine learning.

The project compares Multinomial Naive Bayes and Logistic Regression, evaluates their performance, and saves the final trained model for future predictions.

## Project Overview

Spam messages often contain misleading offers, promotional content, suspicious links, and fraudulent claims. This project aims to identify such messages automatically using machine learning.

The classifier analyzes the text of an SMS message and predicts whether it is spam or legitimate.

**Main objectives:**
- Clean and preprocess SMS text.
- Convert text into numerical features using TF-IDF.
- Train and compare two machine learning algorithms.
- Evaluate performance using classification metrics.
- Select an appropriate decision threshold.
- Save the trained model for future use.
- Predict whether newly entered messages are spam or ham.

## Technologies Used

- Python
- Jupyter Notebook
- Pandas
- NumPy
- NLTK
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib

## Dataset

The project uses the **SMS Spam Collection Dataset**.

Original dataset statistics:

| Category | Messages |
|---|---:|
| Total messages | 5,572 |
| Ham | 4,825 |
| Spam | 747 |

After removing duplicate messages:

| Category | Messages |
|---|---:|
| Total messages | 5,169 |
| Ham | 4,516 |
| Spam | 653 |

The dataset contains two categories:

- **Ham:** Normal and legitimate SMS messages.
- **Spam:** Unwanted, promotional, or potentially fraudulent messages.

## Project Workflow

### 1. Data Loading and Exploration

- Loaded the SMS dataset using Pandas.
- Assigned appropriate column names.
- Checked dataset dimensions, missing values, class distribution, and duplicate records.
- Removed duplicate messages.

### 2. Text Preprocessing

Text preprocessing prepares raw messages for machine learning.

The following operations were performed:

1. Lowercasing: Converted text into lowercase.
2. Tokenization: Split messages into individual words.
3. Stop-word removal: Removed common English words that provide limited classification information.
4. Negation preservation: Important words such as "not", "no", and "don't" were retained because they can change a message's meaning.

### 3. Dataset Splitting

The cleaned dataset was divided into three subsets using stratified sampling.

| Dataset | Percentage | Messages |
|---|---:|---:|
| Training | 70% | 3,618 |
| Validation | 15% | 775 |
| Testing | 15% | 776 |

The validation set was used for model comparison and threshold selection. The test set was kept separate for final evaluation.

### 4. Feature Extraction Using TF-IDF

Machine learning algorithms cannot directly understand raw text.

TF-IDF (Term Frequency-Inverse Document Frequency) converts text into numerical feature vectors.

Configuration:
- Vectorization technique: TF-IDF
- Vocabulary features: 7,052
- Fitted on training data only to prevent data leakage.

### 5. Model Training

Two machine learning algorithms were trained and compared.

**Multinomial Naive Bayes**
- Suitable for text classification.
- Uses word-related probability estimates.
- Performs efficiently on sparse text features.

**Logistic Regression**
- A supervised classification algorithm.
- Estimates the probability of a message belonging to a class.
- Provides a useful comparison with Naive Bayes.

### 6. Model Comparison

Validation results:

| Metric | Naive Bayes | Logistic Regression |
|---|---:|---:|
| Accuracy | 95.35% | 95.10% |
| Spam Precision | 100.00% | 95.45% |
| Spam Recall | 63.27% | 64.29% |
| Spam F1-score | 77.50% | 76.83% |

**Selected Model: Multinomial Naive Bayes**

Naive Bayes was selected because the project prioritizes high spam precision to reduce legitimate messages being incorrectly classified as spam.

### 7. Decision Threshold Optimization

The default classification threshold is 0.50.

Different thresholds were evaluated using the validation set.

| Threshold | Precision | Recall | F1-score |
|---|---:|---:|---:|
| 0.30 | 98.63% | 73.47% | 84.21% |
| 0.35 | 100.00% | 70.41% | 82.63% |
| 0.40 | 100.00% | 68.37% | 81.21% |
| 0.45 | 100.00% | 66.33% | 79.75% |
| 0.50 | 100.00% | 63.27% | 77.50% |

The threshold of **0.30** was selected because it provided the highest validation F1-score among the tested thresholds.

This threshold improves spam detection while maintaining high precision.

### 8. Final Model Evaluation

After selecting the model and threshold, the model was retrained using the combined training and validation datasets.

The final model was evaluated on the untouched test dataset.

**Final Test Results**

| Metric | Score |
|---|---:|
| Accuracy | 97.94% |
| Spam Precision | 98.81% |
| Spam Recall | 84.69% |
| Spam F1-score | 91.21% |

**Confusion Matrix**

| | Predicted Ham | Predicted Spam |
|---|---:|---:|
| Actual Ham | 677 | 1 |
| Actual Spam | 15 | 83 |

Interpretation:

- 677 legitimate messages were correctly classified.
- 83 spam messages were correctly detected.
- 1 legitimate message was incorrectly classified as spam.
- 15 spam messages were incorrectly classified as legitimate.

These results show strong performance on the test dataset. However, performance on new real-world messages may differ.

## Model Saving

The final trained model was saved using Joblib.

**Saved file:**

`spam_classifier_model.pkl`

The saved artifact contains:

- Trained TF-IDF and Multinomial Naive Bayes pipeline.
- Selected decision threshold (0.30).
- Class names.
- Stop-word configuration.

The saved artifact can be loaded later without retraining the model.

## Example Predictions

**Example 1: Spam Message**

Input:

`Congratulations! You won a free iPhone. Click here!`

Output:

```text
Cleaned Message: congratulations free iphone click
Prediction: SPAM
Spam Probability: 41.35%
```

**Example 2: Legitimate Message**

Input:

`Hey, are we meeting tomorrow?`

Output:

```text
Cleaned Message: hey meeting tomorrow
Prediction: HAM
Spam Probability: 0.60%
```

## How to Run the Project

### Requirements

Install the required Python libraries:

```bash
pip install pandas numpy nltk scikit-learn matplotlib seaborn joblib jupyter
```

### Run the Notebook

1. Download or clone this project.
2. Extract the SMS Spam Collection dataset into the project directory.
3. Open the project folder.
4. Start Jupyter Notebook or JupyterLab.
5. Open `Spam_Message_Classifier.ipynb`.
6. Execute the notebook cells in order.

### Load the Saved Model

```python
import joblib

artifact = joblib.load("spam_classifier_model.pkl")

model = artifact["model"]
threshold = artifact["threshold"]
stop_words = artifact["stop_words"]

print("Model loaded successfully!")
```

For new predictions, apply the same preprocessing used during training before passing the cleaned message to the model.

## Project Structure

```text
spam message clasifier/
│
├── dataset/
│   └── SMSSpamCollection
│
├── Spam_Message_Classifier.ipynb
├── spam_classifier_model.pkl
└── README.md
```

## Future Improvements

- Develop a web-based interface using Flask or Streamlit.
- Integrate real-time SMS classification.
- Experiment with additional machine learning algorithms.
- Test on larger and more diverse SMS datasets.
- Improve detection of previously unseen spam patterns.
- Monitor false positives and false negatives using new evaluation data.

## Conclusion

This project demonstrates how Natural Language Processing and machine learning can be used to classify SMS messages automatically.

Multinomial Naive Bayes achieved a test accuracy of 97.94%, with 98.81% spam precision and a 91.21% spam F1-score at the selected 0.30 decision threshold.

The final model was saved using Joblib and successfully tested after loading it again, demonstrating that it can be reused for future SMS classification.

---

**Project Type:** Machine Learning / Natural Language Processing

**Task:** Binary Text Classification

**Final Algorithm:** Multinomial Naive Bayes

**Feature Extraction:** TF-IDF

**Model Format:** Joblib (`.pkl`)