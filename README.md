# Financial Phrasebank Sentiment Analysis
## Goal
Process text data and predict sentiment (positive, neutral, negative) in financial phrasebank using Natural Language Processing Techniques.

## Approach
- Used TF-IDF vectorization and logistic regression for sentiment classification.
- Preprocessed text with tokenization, stopword removal, and lemmatization; explored patterns with word clouds, sentiment distribution, and word count box plots.
- Evaluated models via confusion matrices, classification reports, and SHAP plots for interpretability.

## Key Results
- Best model: Unigram TF-IDF logistic regression with 83% accuracy.
- Positive sentiment recall improved with N-gram (1-5 word) model at the cost of precision.
- Found dataset bias toward neutral sentiment and influence of temporal terms.

## Links
- [Project Jupyter Notebook](https://github.com/nvpham12/Telecom-Customer-Churn-Prediction/blob/main/Telecom_Customer_Churn_Prediction.ipynb)
- [Full Technical Report](https://github.com/nvpham12/Financial-News-Sentiment-Analysis/blob/main/Technical%20Report%20Financial%20Phrasebank%20Sentiment.md)

## Data Source and License
- Dataset: Financial Phrasebank
- Authors: Pekka Malo, Ankur Sinha, Panu Korhonen, Jyrki Wallenius, and Peter Takala  
- Source: [Hugging Face Dataset Repository](https://huggingface.co/datasets/takala/financial_phrasebank)  
- License: Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported (CC BY-NC-SA 3.0)  
- Reference: Malo, P., Sinha, A., Korhonen, P., Wallenius, J., & Takala, P. (2014). *Good debt or bad debt: Detecting semantic orientations in economic texts*. Journal of the Association for Information Science and Technology, 65(4), 782–796.
