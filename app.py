import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("📄 Resume Fit Analyzer")
st.write("Check how well your resume matches a job description")

st.write("--------------------------------------------------")

# Inputs
resume = st.text_area("Paste your resume here", "Python SQL Data Analysis Machine Learning")
job_desc = st.text_area("Paste job description here", "Looking for Python and SQL skills")

# Button
if st.button("Analyze"):

    if resume.strip() and job_desc.strip():

        vectorizer = TfidfVectorizer(stop_words='english')
        vectors = vectorizer.fit_transform([resume, job_desc])

        score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        percent = int(score * 100)

        # Match score
        st.subheader("Match Score")
        st.progress(percent / 100)
        st.write(f"{percent}% match")

        st.write("--------------------------------------------------")

        # Keywords
        resume_words = set(resume.lower().split())
        jd_words = set(job_desc.lower().split())

        missing = jd_words - resume_words

        st.subheader("Missing Keywords")

        if len(missing) > 0:
            st.write(list(missing)[:10])
        else:
            st.write("No major keywords missing")

        st.write("--------------------------------------------------")

        # Suggestions
        st.subheader("Suggestions")

        if percent < 40:
            st.error("Low match. Add more relevant skills and keywords.")
        elif percent < 70:
            st.warning("Decent match. You can improve by adding missing keywords.")
        else:
            st.success("Good match! Your resume fits well.")

    else:
        st.warning("Please enter both fields")