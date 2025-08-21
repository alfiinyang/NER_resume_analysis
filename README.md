# 📄 Resumé Analysis using NER with Cosine Similarity

This project implements a resume analysis pipeline using **Named Entity Recognition (NER)** and **cosine similarity scoring** to aid hiring professionals in quickly filtering candidate profiles by extracting key information and comparing resumes semantically with job specifications.

Read my article [**Performing Resumé Analysis using NER with Cosine Similarity**](https://medium.com/pythons-gurus/performing-resum%C3%A9-analysis-using-ner-with-cosine-similarity-8eb99879cda4) for a deeper dive into the concepts and full walkthrough.

---

## 🚀 Overview

The pipeline—based on the process outlined in [the Medium article](https://medium.com/pythons-gurus/performing-resum%C3%A9-analysis-using-ner-with-cosine-similarity-8eb99879cda4)—follows six essential steps:  
1. Importing necessary packages (spaCy, NLTK, pandas, PyPDF2)  
2. Loading resume PDFs and NER model  
3. Creating custom NER patterns via `EntityRuler`  
4. Cleaning and preprocessing text for accuracy  
5. Performing entity recognition  
6. Computing similarity scores between resumes and reference texts (e.g., job descriptions)

---

## 📂 Project Structure

```

.
├── NER\_resume\_analysis.ipynb     # Main workflow notebook
├── resumes.csv                   # Generated dataset of resume texts
├── skills\_patterns.json          # Custom skill/entity patterns (optional)
├── README.md                     # Project documentation
└── requirements.txt              # Dependencies list

````

---

## 🛠️ Installation

```bash
# Install dependencies
python -m spacy download en_core_web_sm
````

### `requirements.txt` should include:

```
spacy
PyPDF2
pandas
nltk
scikit-learn
```

---

## 📖 Usage

1. **Prepare PDF Resumes**: Place in project directory.
2. **Run Notebook**: Execute `NER_resume_analysis.ipynb` sequentially:

   * **Import packages**: `spaCy`, `NLTK` (`stopwords`, `WordNetLemmatizer`), `PyPDF2`, `pandas`
   * **Extract resume text** and save to `resumes.csv`
   * **Clean text**: Lowercasing, removing punctuation, HTML tags, URLs; lemmatizing using NLTK—enhances NER accuracy
   * **Load spaCy model** and initialize an `EntityRuler` with custom patterns (e.g., technical skills, names)
   * **Run NER** to extract entities like `PERSON`, `SKILL`, `ORG`
   * **Compute similarity**:

     * Use TF-IDF vectorization or embeddings (Doc2Vec, Sentence-BERT) with `cosine_similarity` to measure alignment with a job description

---

## 🧾 Sample Snippets

**Text Cleaning with NLTK:**

```python
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

lemmatizer = WordNetLemmatizer()
clean_text = re.sub(r'http\S+|<.*?>|[^a-zA-Z\s]', '', raw_text.lower())
tokens = [lemmatizer.lemmatize(word) for word in clean_text.split() 
          if word not in stopwords.words('english')]
processed_text = " ".join(tokens)
```

**NER with Custom Patterns:**

```python
import spacy
from spacy.pipeline import EntityRuler

nlp = spacy.load("en_core_web_sm")
ruler = EntityRuler(nlp)
patterns = [{"label": "SKILL", "pattern": "Python"},
            {"label": "SKILL", "pattern": "Pandas"}]
ruler.add_patterns(patterns)
nlp.add_pipe(ruler, before="ner")
```

**Cosine Similarity with TF-IDF:**

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["resume_text"])
job_vec = vectorizer.transform([job_description])
scores = cosine_similarity(job_vec, tfidf_matrix).flatten()
```

---

## 📊 Sample Output

| Resume ID | Entity     | Label  |
| --------- | ---------- | ------ |
| 1         | Ime Inyang | PERSON |
| 1         | Python     | SKILL  |
| 1         | Pandas     | SKILL  |

Additionally, resume-to-job matching scores are generated using cosine similarity metrics (and optionally ranked).

---

## 🔮 Future Improvements

* Expand `skills_patterns.json` with broader technical & soft skills
* Train a **custom NER model** (e.g., transformer-based) for improved accuracy
* Leverage **Sentence-BERT** embeddings for better semantic similarity
* Build a Flask/Dash **web interface** for live upload and scoring
* Integrate **ranking/scoring dashboards** with visualization
* Support dynamic job-role-specific profiling

---

## 🙌 Acknowledgments

* Read my article on [**Performing Resumé Analysis using NER with Cosine Similarity**](https://medium.com/pythons-gurus/performing-resum%C3%A9-analysis-using-ner-with-cosine-similarity-8eb99879cda4)
* Based on foundational NER concepts discussed by [Adib Ali Anwan on DataCamp](https://www.datacamp.com/blog/what-is-named-entity-recognition-ner)
* Similarity techniques reference TF-IDF, Doc2Vec, and BERT-based embeddings

---

## 📜 License

MIT License. Feel free to use, modify, and extend—especially for applications in recruiting, NLP, and resume analytics.
