# Project Overview
This project focuses on predicting sentiment in text from financial news using natural language processing (NLP) and logistic regression. The workflow includes cleaning and preprocessing text data, visualizing sentiment distribution and word count trends, and analyzing word usage across unigrams, bigrams, and n-grams using word clouds. TF-IDF–based logistic regression models are trained to classify sentiment, with performance evaluated through confusion matrices, classification report heatmaps, and SHAP plots to visualize influential words and phrases.

For more technical information and to view the code, please refer to the Jupyter Notebook.

# Tools and Technologies
- **Pandas**: data manipulation and cleaning
- **Matplotlib / Seaborn / Wordcloud**: EDA and visualizations
- **Scikit-learn**: data preprocessing and machine learning modeling
- **NLTK**: text specific data cleaning
- **SHAP**: computing Shapley values and determining feature importance

# Data Source and License
- Dataset: Financial Phrasebank
- Authors: Pekka Malo, Ankur Sinha, Panu Korhonen, Jyrki Wallenius, and Peter Takala  
- Source: [Hugging Face Dataset Repository](https://huggingface.co/datasets/takala/financial_phrasebank)  
- License: Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported (CC BY-NC-SA 3.0)  
- Reference: Malo, P., Sinha, A., Korhonen, P., Wallenius, J., & Takala, P. (2014). *Good debt or bad debt: Detecting semantic orientations in economic texts*. Journal of the Association for Information Science and Technology, 65(4), 782–796.

# Data
- The datasets contains sentences and labels from various financial news sources. 
- Labels take on values of (0, 1, 2) which represent (negative, neutral, positive) and are labeled are annotated by 8 people. 
- The specific dataset used is the all_agree dataset, which includes only sentences and labels where all 8 annotators agree on the same sentiment that the sentence takes.

# Cleaning
- Duplicates are identified and removed.
- Punctuation, whitespace, and stop words are removed from the text.
- Text is converted to lowercase and lemmatized, grouping words into groups based on their roots (ex. increase, increased, and increasing are one group).
- Cleaned text is added as a column to the dataset.

# Exploratory Data Analysis
## Sentiment Distribution
<img width="1000" height="600" alt="sentiment_distribution" src="https://github.com/user-attachments/assets/e7120ebf-8131-44f5-aa89-71141bc13688" />

- It looks like most over half of the sentiment in the data is labeled as neutral. 
- Positive sentiment is around double the negative sentiment. 
- The data is imbalanced and sentiment prediction models are suceptible to biases toward neutral sentiment.
---

## Raw Word Count Distribution
<img width="1000" height="600" alt="word_count_dist_uncleaned" src="https://github.com/user-attachments/assets/3ae837f3-df4f-4625-9a60-34126910e26b" />

- Word count is slightly higher when sentiment is positive vs negative, but the difference is small.
- When sentiment is neutral, word count tends to be the lowest.
- Neutral sentiment also has the most outliers in word count.
- Positive sentiment seems to have a larger box than negative sentiment, indicating larger spread in word count.
- In the context of the data, financial news tends to have longer sentences when sentiment has a clear direction.
- For all sentiment classes, the data seems to be skewed towards higher word counts since the line representing the median is below the center of each box.
---

## Cleaned Word Count Distribution
<img width="1000" height="600" alt="word_count_dist_cleaned" src="https://github.com/user-attachments/assets/4629f6ff-6c1e-4051-96c0-67b1f61822c5" />

- After cleaning the text and removing stopwords, the boxes for each sentiment are now lower. 
- The previous pattern of positive > negative > neutral sentiment in word count is observed again.
- Skew is more obvious for positive and negative sentiment (the median line is closer to the bottom of the box).
- For neutral sentiment, the median line approaches the middle of the box indicating reduced skew.
- The box for positive sentiment appears larger indicating wider spread in word count.

## Word Clouds
Multiple Word Clouds were generated based on N-Gram, representing word count within phrases. Unigrams contain only a single word, while Bigrams are phrases containing 2 words.

### Unigram Word Cloud
<img width="1000" height="600" alt="unigram_wordcloud" src="https://github.com/user-attachments/assets/a8db9292-cc33-4277-a238-bd5afe19d2ac" />

- Financial currencies and business terms appear frequently.
- The data seems to be dominantly based around European companies and/or news since euro appears most frequently, although usd has also appeared in the wordcloud.
- From the size of the words, the news mentions sales, profits, and million most frequently.
- However, with single words it can be difficult to identify sentiment. Words like 'increase' or 'fell' could be used describe to either profits or losses.

---

### Bigram Word Cloud
<img width="1000" height="600" alt="bigram_wordcloud" src="https://github.com/user-attachments/assets/b869fb5c-a95d-4270-b433-69c2343a56cf" />

- With Bigrams, the word cloud contains 2 word phrases instead.
- The biggest change in this word cloud are the temporal anchors and directional signals such as more specific time periods and whether profit increased or decreased.
- Other financial terms like cash flow and voting rights can now be captured.

# Sentiment Prediction Models
Multiple models were built using Term Frequency Inverse Document Frequency (TF-IDF) vectorization combined with a logistic regression classifier. This is a hybrid vectorization method using Term Frequency (TF) and Inverse Document Frequency (IDF). TF measures the number of occurances of a term, while IDF penalizes terms that appear across many documents. TF-IDF combines both methods, emphasizing terms that are frequent but uniquely distributed across different observations.

The TF-IDF vectorizer converts text to numerical features and inputs those into a logistic regression classification algorithm, which finds the probabilities for each sentiment class and chooses the sentiment with the highest probability as its prediction.

Models built include:
- Unigram
- Bigram
- Trigram
- N-Gram (1-5 words)

## Model Performance Metrics
- A classification report is generated for each model, showing model performance metrics such as Precision, Recall, Accuracy, and F1-Score.
- A confusion matrix is generated for each model to show how its predictions compare to the actual sentiment labels.
- SHAP values are computed and visualized to highlight which words or phrases most influence the model’s predictions.
- The confusion matrices and SHAP plots can be found in the Jupyter Notebook.

### Metrics
| Model     | Sentiment | Precision | Recall | F1-Score |
|-----------|-----------|-----------|--------|----------|
| Unigram   | Negative  | 0.62      | 0.65   | 0.64     |
|           | Neutral   | 0.88      | 0.96   | 0.92     |
|           | Positive  | 0.81      | 0.61   | 0.70     |
| Bigram    | Negative  | 0.57      | 0.48   | 0.52     |
|           | Neutral   | 0.85      | 0.96   | 0.90     |
|           | Positive  | 0.73      | 0.56   | 0.63     |
| Trigram   | Negative  | 0.53      | 0.23   | 0.32     |
|           | Neutral   | 0.71      | 1.00   | 0.83     |
|           | Positive  | 0.83      | 0.29   | 0.43     |
| N-Gram    | Negative  | 0.63      | 0.63   | 0.63     |
|           | Neutral   | 0.89      | 0.95   | 0.92     |
|           | Positive  | 0.76      | 0.64   | 0.70     |

- The Unigram and the N-Gram models have the strongest performance, with minimal variation between them.
- Performance declines when the model is trained on longer phrases like bigrams and trigrams.

---

### Accuracy and Macro-Average Metrics
| Model   | Precision | Recall | F1-Score | Accuracy |
|---------|-----------|--------|----------|----------|
| Unigram | 0.77      | 0.74   | 0.75     | 0.83     |
| Bigram  | 0.71      | 0.67   | 0.68     | 0.79     |
| Trigram | 0.69      | 0.51   | 0.53     | 0.71     |
| N-Gram  | 0.76      | 0.74   | 0.75     | 0.83     |
> Note: This table uses macro-averages for precision, recall, and f1-score.

- Results from the previous table are reflected here; the accuracy scores are the same for the Unigram and N-Gram models and macro-averages for precision, recall, and f1-score are about the same.
- The performance decline from unigram to bigram is lower than the decline from bigram to trigram.

# Insights
- In this dataset, unigrams contributed the most to best performing models.
- Allowing the model to check for N-grams, including but not limited to the unigrams, does not affect model performance much for Negative and Neutral sentiment, but does have an observable increase in recall for Positive sentiment.
- Overall model performance declines when the model is trained on longer phrases, with each additional term driving larger performance drops. 
- The N-gram model retains the same level of performance for negative and neutral sentiment in the unigram baseline, while providing a slight boost in Positive recall and clearer SHAP interpretability at the expense of Positive precision.
- The data may contain noise that makes the model biased towards Neutral sentiment. Temporal anchors are suspected to be responsible.

# Next Steps
- Explore potential length bias across sentiment classes by normalizing TF-IDF by document length or including word count as a feature.
- Explore different random seeds and/or resample the data before modeling.
- Increase proportion of training data when splitting, changing the 70/30 split to an 80/20 split. Since the dataset is small, model performance may change by increasing the sample size of the training data.
- Adjust cleaning to exclude certain financial phrases like year-on-year.
- Account for overlapping words when modeling N-grams. This would also require steps to excluded terms like operating profit or net profit from the process, since those are specific financial terms rather than actual overlap.
- Try removing temporal anchors and check for any changes in model performance as those terms may be noise.
- Fit different models using other N-gram ranges such as 1-3 words or 2-5 words and compare the models' performance to the Unigram model.
