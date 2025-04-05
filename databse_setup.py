import sqlite3
import pandas as pd
import datetime
import random
import spacy
from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load NLP model for skill extraction
nlp = spacy.load("en_core_web_sm")

# ----- Connect to SQLite Database -----
DB_NAME = "recruitment.db"
conn = sqlite3.connect(DB_NAME)
cursor = conn.cursor()

# ----- Create Tables -----
cursor.executescript("""
CREATE TABLE IF NOT EXISTS JobDescriptions (
    job_id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_title TEXT,
    job_description TEXT,
    required_skills TEXT
);

CREATE TABLE IF NOT EXISTS Candidates (
    candidate_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    skills TEXT
);

CREATE TABLE IF NOT EXISTS ShortlistedCandidates (
    shortlist_id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id INTEGER,
    candidate_id INTEGER,
    match_score INTEGER,
    FOREIGN KEY (job_id) REFERENCES JobDescriptions(job_id),
    FOREIGN KEY (candidate_id) REFERENCES Candidates(candidate_id)
);

CREATE TABLE IF NOT EXISTS Interviews (
    interview_id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id INTEGER,
    job_id INTEGER,
    interview_date TEXT,
    FOREIGN KEY (candidate_id) REFERENCES Candidates(candidate_id),
    FOREIGN KEY (job_id) REFERENCES JobDescriptions(job_id)
);
""")
conn.commit()


# ----- Skill Extraction Function -----
def extract_skills(text):
    """
    Extracts skills from a job description using NLP.
    Assumes skills are usually nouns or proper nouns.
    """
    doc = nlp(text)
    skills = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN"]]
    return ", ".join(set(skills))  # Remove duplicates


# ----- Load CSV Data into Database -----
jd_df = pd.read_csv(r"C:\Users\91842\Downloads\Dataset\Dataset\[Usecase 5] AI-Powered Job Application Screening System\job_description.csv",encoding="ISO-8859-1")
cv_df = pd.read_csv("Candidates.csv")

# Insert Job Descriptions (with skill extraction)
for _, row in jd_df.iterrows():
    extracted_skills = extract_skills(row["Job Description"])  # Extract skills
    cursor.execute("INSERT INTO JobDescriptions (job_title, job_description, required_skills) VALUES (?, ?, ?)",
                   (row["Job Title"], row["Job Description"], extracted_skills))

# Insert Candidates
for _, row in cv_df.iterrows():
    cursor.execute("INSERT INTO Candidates (name, skills) VALUES (?, ?)",
                   (row["name"], row["skills"]))

conn.commit()
print("\n✅ Data Loaded into Database with Skill Extraction!")


# ----- Matching Function -----
def calculate_match_score(jd_skills, candidate_skills, method="tfidf"):
    if method == "tfidf":
        vectorizer = TfidfVectorizer().fit([jd_skills, candidate_skills])
        vectors = vectorizer.transform([jd_skills, candidate_skills])
        similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
        return int(similarity * 100)

    elif method == "fuzzy":
        return fuzz.ratio(jd_skills, candidate_skills)

    return 0


# ----- Process Matching & Shortlisting -----
SHORTLIST_THRESHOLD = 80

cursor.execute("SELECT * FROM JobDescriptions")
jds = cursor.fetchall()

cursor.execute("SELECT * FROM Candidates")
candidates = cursor.fetchall()

shortlisted = []

for jd in jds:
    job_id, job_title, job_description, jd_skills = jd
    print(f"\n🔍 Matching Candidates for Job: {job_title}")

    for candidate in candidates:
        candidate_id, candidate_name, candidate_skills = candidate

        match_score = calculate_match_score(jd_skills, candidate_skills, method="tfidf")

        print(f"   📝 {candidate_name} → Match Score: {match_score}%")

        if match_score >= SHORTLIST_THRESHOLD:
            shortlisted.append((job_id, candidate_id, match_score))
            cursor.execute("INSERT INTO ShortlistedCandidates (job_id, candidate_id, match_score) VALUES (?, ?, ?)",
                           (job_id, candidate_id, match_score))

conn.commit()
print("\n✅ Shortlisting Completed!")

# ----- Interview Scheduling -----
today = datetime.date.today()

for job_id, candidate_id, match_score in shortlisted:
    interview_date = today + datetime.timedelta(days=random.randint(1, 14))
    cursor.execute("INSERT INTO Interviews (candidate_id, job_id, interview_date) VALUES (?, ?, ?)",
                   (candidate_id, job_id, interview_date.strftime('%Y-%m-%d')))

conn.commit()
print("\n📅 Interviews Scheduled!")

# Close Database Connection
conn.close()
