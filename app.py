import streamlit as st
import re
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

st.title("📄 Resume Fit Analyzer")
st.write("A simple tool to check how well your resume matches a job description")

st.write("--------------------------------------------------")

# Clean text function
def clean_text(text):
    text = text.lower()

    # handle important phrases first
    text = text.replace("deep learning", "deeplearning")
    text = text.replace("machine learning", "machinelearning")
    text = text.replace("problem-solving", "problemsolving")

    # remove symbols
    text = re.sub(r'[^a-zA-Z ]', '', text)

    words = text.split()

    # custom stopwords
    custom_stopwords = {
        "looking", "skills", "skill", "experience", "knowledge", "candidate"
    }

    words = [
        w for w in words
        if w not in ENGLISH_STOP_WORDS and w not in custom_stopwords
    ]

    return set(words)

# Inputs
resume = st.text_area(
    "Paste your resume here",
    "Python, SQL, Data Analysis, Pandas, NumPy, Machine Learning"
)

job_desc = st.text_area(
    "Paste job description here",
    "Looking for Python, SQL, Machine Learning, Deep Learning, NLP, Data Visualization, Statistics, and problem-solving skills"
)

# Button
if st.button("Analyze"):

    if resume.strip() and job_desc.strip():

        # TF-IDF similarity
        vectorizer = TfidfVectorizer(stop_words='english')
        vectors = vectorizer.fit_transform([resume, job_desc])

        score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        percent = int(score * 100)

        # Match score
        st.subheader("Match Score")
        st.progress(percent / 100)
        st.write(f"{percent}% match")

        st.write("--------------------------------------------------")

        # Cleaned keyword comparison
        resume_words = clean_text(resume)
        jd_words = clean_text(job_desc)

        missing = jd_words - resume_words

        st.subheader("Missing Keywords")

        if len(missing) > 0:
            # convert back to readable form
            display_words = [
                w.replace("deeplearning", "deep learning")
                 .replace("machinelearning", "machine learning")
                 .replace("problemsolving", "problem-solving")
                for w in list(missing)[:10]
            ]

            st.write(display_words)
        else:
            st.write("No major keywords missing, but overall similarity is still low")

        st.write("--------------------------------------------------")

        # Suggestions
        st.subheader("Suggestions")

        if percent < 40:
            st.error("Low match. Try adding more relevant skills.")
        elif percent < 70:
            st.warning("Decent match. You can improve by adding missing keywords.")
        else:
            st.success("Good match! Your resume fits well.")

    else:
        st.warning("Please enter both fields")