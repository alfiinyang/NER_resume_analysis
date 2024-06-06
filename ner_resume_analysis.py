!pip install PyPDF2

import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import PyPDF2
import pandas as pd

def extract_text_from_pdf(pdf_path):
    text = ''
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

# List of PDF file paths containing resumes
pdf_files = ['resume1.pdf', 'resume2.pdf', 'resume3.pdf', 'resume4.pdf', 'resume5.pdf'] # replace resumes with filepaths to resumes for the analysis

# Extract text from each PDF resume and store it in a list
resumes_text = [extract_text_from_pdf(pdf_path) for pdf_path in pdf_files]

# Create a DataFrame with columns 'ID' and 'resume_text'
data = pd.DataFrame({'ID': range(1, len(pdf_files)+1), 'resume_text': resumes_text})

# Save the DataFrame to a CSV file
data.to_csv('resumes.csv', index=False)

# Load data from CSV file
data = pd.read_csv('resumes.csv')  # Replace 'resumes.csv' with the path to your CSV file

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

from spacy.pipeline import EntityRuler

# Add entity ruler pipeline to spaCy model
ruler = nlp.add_pipe("entity_ruler", before="ner")

# Define patterns as dictionaries
patterns = [
    {"label": "SKILL", "pattern": [{"LOWER": "skill_1"}]},
    {"label": "SKILL", "pattern": [{"LOWER": "skill_2"}]},
    {"label": "SKILL", "pattern": [{"LOWER": "skill_3"}]},
    {"label": "SKILL", "pattern": [{"LOWER": "skill_4"}]}
]

# Add patterns to entity ruler
ruler.add_patterns(patterns)

import re
from nltk.tokenize import word_tokenize

# Initialize WordNet Lemmatizer
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    # Remove hyperlinks, special characters, and punctuations using regex
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s]', '', text)

    # Convert the text to lowercase
    text = text.lower()

    # Tokenize the text using nltk's word_tokenize
    words = word_tokenize(text)

    # Lemmatize the text to its base form for normalization
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]

    # Remove English stop words
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in lemmatized_words if word not in stop_words]

    return filtered_words

# Clean the 'resume_text' column in the DataFrame
data['cleaned_resume'] = data['resume_text'].apply(clean_text)

from spacy import displacy

# Define options for visualization
options = {'ents': ['PERSON', 'GPE', 'SKILL'], 'colors': {'PERSON': 'orange', 'GPE': 'lightgreen', 'SKILL': 'lightblue'}}

# Visualize named entities in each resume
for resume_text in data['resume_text']:
    doc = nlp(resume_text)
    displacy.render(doc, style="ent", jupyter=True, options=options)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Define the company requirements
company_requirements = "Company Requirement"

# Combine the company requirements with stopwords removed
cleaned_company_requirements = clean_text(company_requirements)
cleaned_company_requirements_str = ' '.join(cleaned_company_requirements)

# Calculate TF-IDF vectors for the company requirements and resume texts
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(data['resume_text'])
company_tfidf = tfidf_vectorizer.transform([cleaned_company_requirements_str])

# Calculate cosine similarity between the company requirements and each resume
similarity_scores = cosine_similarity(company_tfidf, tfidf_matrix).flatten()

# Get the indices of resumes sorted by similarity score
sorted_indices = similarity_scores.argsort()[::-1]

# Display the top 5 most similar resumes
top_n = 5
for i in range(top_n):
    index = sorted_indices[i]
    print(f"Resume ID: {data['ID'][index]}")
    print(f"Similarity Score: {similarity_scores[index]}")
    print(data['resume_text'][index])
    print()

def calculate_similarity(resume_text, required_skills):
    # Process the resume text with the spaCy model
    doc = nlp(resume_text)

    # Extract skills from the resume using the entity ruler
    skills = [ent.text.lower() for ent in doc.ents if ent.label_ == "SKILL"]

    # Calculate the number of matching skills with required skills
    matching_skills = [skill for skill in skills if skill in required_skills]
    num_matching_skills = len(matching_skills)

    # Calculate the similarity score
    similarity_score = num_matching_skills / max(len(required_skills), len(skills))

    return similarity_score

# Example usage:
for text in data[['cleaned_resume']].itertuples(index = False):
  resume_text = str(text[0])
  print(resume_text)
  required_skills = ["matplotlib", "numpy", "pandas", "data visiualization", "seaborn"]
  similarity_score = calculate_similarity(resume_text, required_skills)
  print("Similarity Score:", similarity_score)
