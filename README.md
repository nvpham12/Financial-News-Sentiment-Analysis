# Project Overview
This project focuses on predicting sentiment in text from financial news using natural language processing (NLP) and logistic regression. The workflow includes cleaning and preprocessing text data, visualizing sentiment distribution and word count trends, and analyzing word usage across unigrams, bigrams, and n-grams using word clouds. TF-IDF–based logistic regression models are trained to classify sentiment, with performance evaluated through confusion matrices, classification report heatmaps, and SHAP plots to visualize influential words and phrases.

For more technical information and to view the code, please refer to the Jupyter Notebook file.

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

- It looks like most over half of the sentiment in the data is labeled as neutral. 
- Positive sentiment is around double the negative sentiment. 
- The data is imbalanced and sentiment prediction models are suceptible to biases toward neutral sentiment.

## Raw Word Count Distribution

- Word count is slightly higher when sentiment is positive vs negative, but the difference is small.
- When sentiment is neutral, word count tends to be the lowest.
- Neutral sentiment also has the most outliers in word count.
- Positive sentiment seems to have a larger box than negative sentiment, indicating larger spread in word count.
- In the context of the data, financial news tends to have longer sentences when sentiment has a clear direction.
- For all sentiment classes, the data seems to be skewed towards higher word counts since the line representing the median is below the center of each box.

## Cleaned Word Count Distribution

- After cleaning the text and removing stopwords, the boxes for each sentiment are now lower. 
- The previous pattern of positive > negative > neutral in word count is observed again.
- Skew seems to become more obvious for positive and negative sentiment (the median line is closer to the bottom of the box).
- For neutral sentiment, the median line approaches the middle of the box indicating reduced skew.
- The box for positive sentiment appears larger indicating wider spread in word count.

## Word Clouds
Multiple Word Clouds were generated based on N-Gram, representing word count within phrases. Unigrams contain only a single word, while Bigrams are phrases containing 2 words.

### Unigram Word Cloud

- Financial currencies and business terms appear frequently.
- The data seems to be dominantly based around European companies and/or news since euro appears most frequently, although usd has also appeared in the wordcloud.
- From the size of the words, the news mentions sales, profits, and million most frequently.
- However, with single words it can be difficult to identify sentiment. Words like 'increase' or 'fell' could be used describe to either profits or losses.

## Bigram Word Cloud

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

## Model Comparison


# Insights
- In this dataset, unigrams contributed the most to best performing models.
- Allowing the model to check for N-grams, including but not limited to the unigrams, does not affect model performance much for Negative and Neutral sentiment, but does have an observable increase in recall for Positive sentiment.
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