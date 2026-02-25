

# 🎬 Movie Recommendation System (Google Colab)

A simple movie recommendation system built in **Python** using **TF-IDF** and **Cosine Similarity**.
Designed for beginners to learn **NLP** and **content-based recommendation systems**.



## 🧰 Features

* Search for a movie by its title
* Get **top N recommended movies** based on plot similarity
* Uses **TF-IDF** to convert text to numerical vectors
* Uses **Cosine Similarity** to find similar movies

---

## 📂 Colab Usage

1. Open the notebook `recommendation.ipynb` in Google Colab.
2. Upload your **movies dataset** (`movies.csv`) to Colab.
3. Run each cell **top to bottom**.
4. Call the function:

```python
recommend("The Dark Knight", top_n=5)
```

---

## 📝 Dataset

* You need a CSV file with at least these columns:

  * `title` → Movie title
  * `overview` → Movie description

* You can use the [TMDB Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)

* Make sure to **upload the CSV to Colab** before running the notebook.

---

## ⚡ How It Works (Step by Step)

1. **Load Data**

```python
import pandas as pd

movies = pd.read_csv("movies.csv")
movies['overview'] = movies['overview'].fillna('')
```

2. **Convert Text to TF-IDF**

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['overview'])
```

3. **Compute Cosine Similarity**

```python
from sklearn.metrics.pairwise import cosine_similarity

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
```

4. **Build Movie Index**

```python
indices = pd.Series(movies.index, index=movies['title']).to_dict()
```

5. **Create Recommendation Function**

```python
def recommend(title, top_n=5):
    if title not in indices:
        return "Movie not found"
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices]
```

---

## 💡 Example

```python
recommend("The Dark Knight", top_n=5)
```

Output:

```
1. Batman Begins
2. Joker
3. Inception
4. The Dark Knight Rises
5. Logan
```

---

## 🚀 Tips for Colab Users

* Upload the CSV using **Colab’s file upload tool**
* Use smaller datasets if TF-IDF takes too long
* Fill missing overviews with empty strings using `.fillna('')`

---

## 📌 References

* [Scikit-learn TF-IDF](https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting)
* [Cosine Similarity Explanation](https://en.wikipedia.org/wiki/Cosine_similarity)
* [TMDB Movie Dataset on Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)


