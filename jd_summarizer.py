

import re
import random
import datetime
import spacy
import nltk
from fuzzywuzzy import fuzz
import pdfplumber
import fitz  # PyMuPDF

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from transformers import pipeline

# Download NLTK resources if not already downloaded
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Load spaCy English model and NLTK stop words
nlp = spacy.load('en_core_web_sm')
stop_words = set(stopwords.words('english'))

# Initialize a transformers summarization pipeline
summarizer_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")

# ----- Helper Functions for PDF Extraction -----
def extract_text_from_pdf(file_path):
    """
    Extracts text from a PDF using pdfplumber, and if that fails, falls back to PyMuPDF.
    """
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        if text.strip():
            return text
    except Exception as e:
        print(f"pdfplumber failed: {e}")

    # Fallback to PyMuPDF (fitz)
    try:
        doc = fitz.open(file_path)
        for page in doc:
            text += page.get_text() + "\n"
        return text
    except Exception as e:
        print(f"PyMuPDF failed: {e}")
    return None

# ----- Job Description Summarizer Agent -----
class JDSummarizerAgent:
    def __init__(self, use_transformers=True, summary_word_threshold=100):
        self.sections = ["Responsibilities", "Qualifications", "Skills", "Experience"]
        self.use_transformers = use_transformers
        self.summary_word_threshold = summary_word_threshold

    def summarize(self, jd_text):
        """
        Extracts key sections from the JD, cleans them using spaCy,
        and, if the section is lengthy, produces a transformer-based summary.
        """
        summary = {}
        for section in self.sections:
            raw_text = self._extract_section(jd_text, section)
            # Use spaCy to clean and lemmatize the text
            doc = nlp(raw_text)
            cleaned_text = " ".join([token.lemma_ for token in doc if not token.is_stop])
            final_text = cleaned_text.strip()
            # If text is long and transformer summarization is enabled, summarize further.
            if self.use_transformers and len(final_text.split()) > self.summary_word_threshold:
                try:
                    transformer_summary = summarizer_pipeline(final_text, max_length=60, min_length=30, do_sample=False)
                    final_text = transformer_summary[0]['summary_text']
                except Exception as e:
                    print(f"Transformers summarization failed: {e}")
            summary[section.lower()] = final_text
        return summary

    def _extract_section(self, text, section_name):
        """
        Extracts text from a section in the JD based on the section header.
        """
        pattern = rf"{section_name}:(.*?)(?=\n[A-Z][a-zA-Z ]+:|\Z)"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return "Not specified"

# ----- Recruiting Agent -----
class RecruitingAgent:
    def extract_candidate_data(self, cv_text):
        """
        Extracts candidate details using regular expressions.
        """
        candidate_data = {}
        candidate_data['education'] = self._extract_field(cv_text, "Education")
        candidate_data['experience'] = self._extract_field(cv_text, "Experience")
        candidate_data['skills'] = self._extract_field(cv_text, "Skills")
        return candidate_data

    def _extract_field(self, text, field_name):
        pattern = rf"{field_name}:(.*?)(?=\n[A-Z][a-zA-Z ]+:|\Z)"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return "Not specified"

    def calculate_match_score(self, jd_summary, candidate_data, method="tfidf"):
        """
        Calculates a match score between candidate skills and JD skills.
        Two methods are provided:
          - 'tfidf': Uses Scikit-learn's TF-IDF vectorizer and cosine similarity.
          - 'fuzzy': Uses FuzzyWuzzy's fuzzy matching.
        """
        jd_skills_text = jd_summary.get('skills', "")
        candidate_skills_text = candidate_data.get('skills', "")

        if method == "tfidf":
            vectorizer = TfidfVectorizer().fit([jd_skills_text, candidate_skills_text])
            vectors = vectorizer.transform([jd_skills_text, candidate_skills_text])
            similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
            # Normalize to a percentage
            return int(similarity * 100)
        elif method == "fuzzy":
            # Tokenize and remove stop words using NLTK
            jd_skills_tokens = [word.lower() for word in word_tokenize(jd_skills_text) if word.lower() not in stop_words]
            candidate_skills_tokens = [word.lower() for word in word_tokenize(candidate_skills_text) if word.lower() not in stop_words]

            if not candidate_skills_tokens or not jd_skills_tokens:
                return 0

            total_score = 0
            comparisons = 0
            for c_skill in candidate_skills_tokens:
                for jd_skill in jd_skills_tokens:
                    score = fuzz.ratio(c_skill, jd_skill)
                    total_score += score
                    comparisons += 1

            average_score = (total_score / comparisons) if comparisons else 0
            normalized_score = min(average_score, 100)
            return int(normalized_score)
        else:
            return 0

# ----- Shortlisting Agent -----
class ShortlistingAgent:
    def __init__(self, threshold=80):
        self.threshold = threshold
        self.shortlisted = []

    def shortlist(self, candidate_name, match_score):
        if match_score >= self.threshold:
            self.shortlisted.append((candidate_name, match_score))
            return True
        return False

    def get_shortlisted_candidates(self):
        return self.shortlisted

# ----- Interview Scheduler Agent -----
class InterviewSchedulerAgent:
    def schedule_interview(self, candidate_name):
        today = datetime.date.today()
        future_date = today + datetime.timedelta(days=random.randint(1, 14))
        invitation = (f"Hello {candidate_name},\n"
                      f"Congratulations! You have been shortlisted for an interview. "
                      f"Please join us on {future_date.strftime('%A, %B %d, %Y')}.\n"
                      "Looking forward to meeting you!\n")
        return invitation

# ----- Chatbot Interface -----
class RecruitmentChatbot:
    def __init__(self):
        self.jd_agent = JDSummarizerAgent()
        self.recruiting_agent = RecruitingAgent()
        self.shortlisting_agent = ShortlistingAgent()
        self.scheduler_agent = InterviewSchedulerAgent()
        self.jd_summary = None

    def run(self):
        print("Welcome to the Recruitment Chatbot!")
        while True:
            print("\nOptions:")
            print("1. Input Job Description (Text or PDF)")
            print("2. Process Candidate CV (Text or PDF)")
            print("3. View Shortlisted Candidates")
            print("4. Schedule Interview for a Candidate")
            print("5. Exit")
            choice = input("Select an option (1-5): ")

            if choice == '1':
                self.input_job_description()
            elif choice == '2':
                self.process_candidate_cv()
            elif choice == '3':
                self.view_shortlisted()
            elif choice == '4':
                self.schedule_interview()
            elif choice == '5':
                print("Exiting chatbot. Goodbye!")
                break
            else:
                print("Invalid option. Please try again.")

    def input_job_description(self):
        print("\nEnter the job description text or type 'file' to provide a PDF file path:")
        choice = input("Your input: ")
        if choice.lower() == 'file':
            file_path = input("Enter the PDF file path: ")
            jd_text = extract_text_from_pdf(file_path)
            if not jd_text:
                print("Failed to extract text from PDF. Please try again.")
                return
        else:
            print("Enter the job description text (end with an empty line):")
            lines = [choice]
            while True:
                line = input()
                if line == "":
                    break
                lines.append(line)
            jd_text = "\n".join(lines)

        self.jd_summary = self.jd_agent.summarize(jd_text)
        print("\nJob Description Summary:")
        for key, value in self.jd_summary.items():
            print(f"\n{key.capitalize()}:\n{value}")

    def process_candidate_cv(self):
        if self.jd_summary is None:
            print("Please input a job description first (Option 1).")
            return
        candidate_name = input("\nEnter candidate name: ")
        print("Enter the candidate's CV text or type 'file' to provide a PDF file path:")
        choice = input("Your input: ")
        if choice.lower() == 'file':
            file_path = input("Enter the PDF file path: ")
            cv_text = extract_text_from_pdf(file_path)
            if not cv_text:
                print("Failed to extract text from PDF. Please try again.")
                return
        else:
            print("Enter the candidate's CV text (end with an empty line):")
            lines = [choice]
            while True:
                line = input()
                if line == "":
                    break
                lines.append(line)
            cv_text = "\n".join(lines)

        candidate_data = self.recruiting_agent.extract_candidate_data(cv_text)
        # Choose matching method: "tfidf" or "fuzzy"
        match_score = self.recruiting_agent.calculate_match_score(self.jd_summary, candidate_data, method="tfidf")
        print(f"\nCandidate '{candidate_name}' has a match score of: {match_score}")
        if self.shortlisting_agent.shortlist(candidate_name, match_score):
            print("Candidate shortlisted!")
        else:
            print("Candidate did not meet the shortlisting threshold.")

    def view_shortlisted(self):
        shortlisted = self.shortlisting_agent.get_shortlisted_candidates()
        if not shortlisted:
            print("No candidates have been shortlisted yet.")
        else:
            print("\nShortlisted Candidates:")
            for name, score in shortlisted:
                print(f"- {name} (Score: {score})")

    def schedule_interview(self):
        candidate_name = input("\nEnter the candidate name for scheduling an interview: ")
        shortlisted = self.shortlisting_agent.get_shortlisted_candidates()
        if any(candidate_name == name for name, _ in shortlisted):
            invitation = self.scheduler_agent.schedule_interview(candidate_name)
            print("\nInterview Invitation:")
            print(invitation)
        else:
            print("Candidate not found in shortlisted list.")

# ----- Run the Chatbot -----
if __name__ == "__main__":
    chatbot = RecruitmentChatbot()
    chatbot.run()
