<p align="center">
  <img src="https://img.shields.io/badge/NLTK%20Library-Natural%20Language%20Processing-9C27B0?style=for-the-badge&logo=python&logoColor=white" alt="NLTK" />
</p>

<h1 align="center">🧠 NLTK – Powering Natural Language Processing in Python</h1>

<p align="center">
  Tokenize • Analyze • Understand
</p>

---

## **1. Introduction**

* **NLTK (Natural Language Toolkit):** A library for text processing.

* Used for:

  * Tokenization
  * Stopword removal
  * Lemmatization & Stemming
  * POS tagging
  * Sentiment analysis

* **Installation:**

```bash
pip install nltk
```

* Download datasets:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
```

---

## **2. Tokenization**

```python
from nltk.tokenize import word_tokenize, sent_tokenize

text = "Hello world! This is NLTK."
print(word_tokenize(text))
print(sent_tokenize(text))
```

---

## **3. Stopword Removal**

```python
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
words = word_tokenize("This is a simple example")
filtered = [w for w in words if w.lower() not in stop_words]
print(filtered)
```

---

## **4. Stemming**

```python
from nltk.stem import PorterStemmer

ps = PorterStemmer()
words = ["running", "runs", "runner"]
print([ps.stem(w) for w in words])
```

---

## **5. Lemmatization**

```python
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
words = ["better", "running", "feet"]
print([lemmatizer.lemmatize(w, pos="v") for w in words])
```

---

## **6. Part-of-Speech Tagging**

```python
from nltk import pos_tag

text = word_tokenize("The quick brown fox jumps over the lazy dog")
print(pos_tag(text))
```

---

## **7. Named Entity Recognition (NER)**

```python
from nltk import ne_chunk
nltk.download('maxent_ne_chunker')
nltk.download('words')

sentence = "Barack Obama was born in Hawaii."
print(ne_chunk(pos_tag(word_tokenize(sentence))))
```

---

## **8. Sentiment Analysis (Basic)**

```python
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()
print(sia.polarity_scores("I love this product!"))
```

---

# **NLTK Practice Questions**

### Beginner

1. Tokenize a sentence into words and sentences.
2. Remove stopwords from a given text.
3. Perform stemming on a list of words.

### Intermediate

1. Lemmatize a paragraph of text.
2. Extract nouns and verbs using POS tagging.
3. Perform sentiment analysis on 5 sample reviews.

### Advanced

1. Build a basic NER pipeline.
2. Clean and preprocess a large text dataset.
3. Create a word frequency counter (excluding stopwords).

---

# **NLTK Mini Projects**

### 1. **Sentiment Analysis Tool**

* Input: User reviews
* Features:

  * Tokenization
  * Stopword removal
  * Sentiment classification (positive/negative/neutral)

---

### 2. **Resume Keyword Extractor**

* Input: Multiple resumes (text)
* Features:

  * Extract skills using POS tagging
  * Generate keyword frequency report

---

### 3. **Fake News Detector**

* Dataset: News articles
* Features:

  * Clean text using NLTK
  * Train classifier (Naive Bayes/Logistic Regression)
  * Predict fake vs real

---

### 4. **Chatbot Preprocessing**

* Features:

  * Tokenize user queries
  * Remove stopwords
  * Lemmatize and clean data for chatbot model

---

### 5. **Text Summarizer**

* Input: Long article
* Features:

  * Tokenize into sentences
  * Score sentences by frequency
  * Extract top 5 sentences as summary

