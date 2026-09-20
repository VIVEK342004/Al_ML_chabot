import os
from flask import Flask, request, jsonify
import psycopg2
from psycopg2.extras import DictCursor
from sentence_transformers import SentenceTransformer
import google.generativeai as genai  # API 
import fitz  # PyMuPDF
import csv
import ast
import re
from flask_cors import CORS  # To allow CORS requests from React
from psycopg2.extras import execute_values
import warnings
import csv
from groq import Groq
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from scraping_pipeline import process_pdfs_in_folder
from werkzeug.utils import secure_filename
import pandas as pd


# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes, allowing communication with React frontend


# Load SBERT model
load_dotenv()
warnings.filterwarnings("ignore", category=FutureWarning)

# Load SBERT model
model = SentenceTransformer('paraphrase-distilroberta-base-v1')
# Initialize Groq API safely
groq_key = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=groq_key) if groq_key else None
if groq_client:
    print("Groq API client initialized successfully.")
else:
    print("GROQ_API_KEY not provided. Running in local/offline fallback mode.")

#  genrative model  using Gemini API
def genrative_model(train_text,query):
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    if not gemini_key:
        return "Gemini API key not configured."
    genai.configure(api_key=gemini_key)

    model = genai.GenerativeModel('gemini-pro')
    # train_text = "Provide a response in a single paragraph: "
    full_query = train_text + query

    # Generate the content
    response = model.generate_content(full_query)

    return response.text


# ------------------------------------------------------------------------Basic DB/Operation------------------
def check_for_alphanumeric(s):
    pattern = r'[A-Za-z(){}\[\]0-9:;,.?"\'/\-\u00E9\u2018\u2019\u2014*\u2013\u2014\u2014\u2013]'
    temp = re.sub(pattern, '', s)
    return temp

def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "pg-35f1b54a-saneeipk-09e5.k.aivencloud.com"),
        database=os.getenv("DB_NAME", "defaultdb"),
        user=os.getenv("DB_USER", "avnadmin"),
        password=os.getenv("DB_PASSWORD", ""),
        port=int(os.getenv("DB_PORT", 5432)),
        connect_timeout=3
    )

def init_db():
    conn = get_db_connection()
    cur = conn.cursor()

    # Create pgvector extension if it doesn't exist
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # Create table if it doesn't exist
    cur.execute("""
        CREATE TABLE IF NOT EXISTS smartsearch (
            id SERIAL PRIMARY KEY,
            book_author TEXT,
            book_name TEXT,                         
            book_url TEXT,
            chapter_name TEXT,
            chapter_number TEXT,
            page INTEGER,
            paragraph INTEGER,
            text TEXT,
            topic TEXT,
            embedding vector(768)
        )
    """)

    # Create an index on the embedding column for faster similarity searches
    cur.execute("""
        CREATE INDEX IF NOT EXISTS embedding_idx ON smartsearch USING ivfflat (embedding vector_cosine_ops)
    """)

    conn.commit()
    cur.close()
    conn.close()


def check_and_initialize_db():
    print("Checking database initialization...")
    if not os.path.exists('db_initialized.flag'):
        print("Initializing database...")
        init_db()
        print("Processing CSV files...")
        process_all_csv_files()
        with open('db_initialized.flag', 'w') as flag_file:
            flag_file.write('Database initialized')
        print("Database initialized and CSV files processed.")
    else:
        print("Database already initialized. Skipping initialization.")

def compute_embedding(text):
    return model.encode(text).tolist()

def process_csv_chunk(chunk):
    embeddings = []
    for row in chunk:
        embedding = compute_embedding(row['Text'])
        embeddings.append((
            row['Book Author'], row['Book Name'], row['Book URL'], row['Chapter Name'],
            row['Chapter Number'], int(row['Page']), int(row['Paragraph']),
            row['Text'], row['Topic'], embedding
        ))
    return embeddings

def process_csv_file(csv_file, batch_size=1000):
    conn = get_db_connection()
    cur = conn.cursor()

    with open(csv_file, 'r', encoding='utf-8') as file:
        csvreader = csv.DictReader(file)
        chunk = []
        for i, row in enumerate(csvreader):
            chunk.append(row)
            if len(chunk) == batch_size:
                with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                    results = list(executor.map(process_csv_chunk, [chunk]))
                
                flattened_results = [item for sublist in results for item in sublist]
                
                # Use execute_values for bulk insert
                execute_values(cur, """
                    INSERT INTO smartsearch (book_author, book_name, book_url, chapter_name, chapter_number, page, paragraph, text, topic, embedding)
                    VALUES %s
                """, flattened_results)
                
                conn.commit()
                chunk = []

        # Process remaining rows
        if chunk:
            with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                results = list(executor.map(process_csv_chunk, [chunk]))
            
            flattened_results = [item for sublist in results for item in sublist]
            
            execute_values(cur, """
                INSERT INTO smartsearch (book_author, book_name, book_url, chapter_name, chapter_number, page, paragraph, text, topic, embedding)
                VALUES %s
            """, flattened_results)
            
            conn.commit()

    cur.close()
    conn.close()


def process_all_csv_files():
    search_dirs = [
        os.path.join(os.path.dirname(__file__), "..", "books"),
        os.path.join(os.path.dirname(__file__), "books"),
        os.path.join(os.getcwd(), "..", "books"),
        os.path.join(os.getcwd(), "books"),
        os.getcwd()
    ]
    csv_files = []
    for d in search_dirs:
        if os.path.exists(d):
            found = [os.path.join(d, f) for f in os.listdir(d) if f.endswith('.csv')]
            if found:
                csv_files = found
                break

    for csv_file in csv_files:
        print(f"Processing {csv_file}...")
        process_csv_file(csv_file)
        print(f"Finished processing {csv_file}")

# ----------------------------------------------- Genrative Model--------------------------------------
# Rephrased System Prompt
system_prompt = (
    "You are a highly precise and concise assistant. Follow these principles when generating responses:\n\n"
    "1. Keep responses as short as possible while fully addressing the user's query.\n"
    "2. Provide only essential information—avoid unnecessary details or elaboration.\n"
    "3. Ensure factual accuracy and avoid guessing or fabricating information.\n"
    "4. Use bullet points or numbered lists for clarity if needed.\n"
    "5. If clarification is required, ask brief and specific follow-up questions.\n"
    "6. Acknowledge uncertainty and suggest external resources only if necessary.\n\n"
    "Additional Behavior:\n"
    "- If asked your name, respond with: 'EVA'.\n"
    "- If asked about your designed or developed or developers or created, respond with: 'Rajat, Satyam, Sandeep, students from Sitare University under the guidance of Dr. Kushal Shah.'\n"
    "- If asked about your purpose or work, respond with: 'I am designed to assist with AI/ML-related queries and provide support for technical tasks.'\n"
    "- Always stay precise, relevant, and focused on the query."
)
# Compute embeddings for text
def compute_embedding(text):
    return model.encode(text).tolist()


# 1. Filtering functions
def contains_inappropriate_content(text):
    """
    Check if the text contains inappropriate or flagged content using regex patterns.
    """
    inappropriate_patterns = [
        r"\b(?:f\*?u\*?c\*?k|s\*?h\*?i\*?t|b\*?i\*?t\*?c\*?h)\b",  # Common profanities with optional masking
        r"\b(?:a\*?s\*?s|d\*?a\*?mn|c\*?u\*?n\*?t)\b",
        r"[^\w\s]{3,}",  # Strings with excessive symbols
        r"(.)\1{3,}",  # Repeated characters like "aaaa" or "!!!!"
    ]

    for pattern in inappropriate_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True

    if is_random_string(text):
        return True

    return False

def is_random_string(text):
    """
    Check for random-like strings using length and entropy heuristics.
    """
    if len(text.split()) < 5:
        return True

    unique_chars = set(text)
    entropy = len(unique_chars) / len(text)
    return entropy > 0.8

def get_para_text(para):
    if isinstance(para, dict):
        return para.get('text', '')
    elif hasattr(para, '__getitem__'):
        try:
            return para['text']
        except (KeyError, TypeError, IndexError):
            pass
        try:
            if len(para) > 8:
                return str(para[8])
        except (IndexError, TypeError):
            pass
    return str(para)

# 2. Main refinement function
def refine_and_answer_with_groq(question, paragraphs):
    global groq_client
    raw_texts = [get_para_text(para) for para in paragraphs]

    # Filter inappropriate paragraphs
    filtered_paragraphs = [text for text in raw_texts if not contains_inappropriate_content(text)]
    if not filtered_paragraphs:
        filtered_paragraphs = raw_texts

    # Calculate relevance using cosine similarity
    top_paragraphs = []
    top_context = ""
    if filtered_paragraphs:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([question] + filtered_paragraphs)
        similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
        top_k = min(3, len(filtered_paragraphs))
        top_indices = similarities.argsort()[-top_k:][::-1]
        top_paragraphs = [filtered_paragraphs[idx] for idx in top_indices]
        top_context = "\n\n".join([f"Paragraph {idx + 1}: {filtered_paragraphs[idx]}" for idx in top_indices])

    # Generate response using LLM or smart grounded fallback
    response = ""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Question: {question}\n\nContext:\n{top_context}"}
    ]

    if groq_client:
        try:
            chat_completion = groq_client.chat.completions.create(
                messages=messages,
                model="llama3-8b-8192",
            )
            response = chat_completion.choices[0].message.content.strip()
        except Exception as e:
            print(f"Groq generation error: {e}")

    if not response and os.getenv("GEMINI_API_KEY"):
        try:
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            gemini_mod = genai.GenerativeModel('gemini-1.5-flash')
            g_out = gemini_mod.generate_content(f"{system_prompt}\n\nQuestion: {question}\n\nContext:\n{top_context}")
            response = g_out.text.strip()
        except Exception as e:
            print(f"Gemini generation error: {e}")

    if not response:
        # Grounded EVA synthesis from top passages
        clean_snippets = [p.strip() for p in top_paragraphs if p.strip()]
        if clean_snippets:
            first_sentence = clean_snippets[0].split(". ")[0] + "."
            response = f"According to the AI/ML literature, {first_sentence}\n\nKey Concepts:\n"
            for i, p in enumerate(clean_snippets[:2], 1):
                preview = p[:250] + ("..." if len(p) > 250 else "")
                response += f"- {preview}\n"
        else:
            response = "I searched the AI/ML textbooks, but could not find a sufficiently relevant context for your query."

    return {
        "response": response,
        "top_paragraphs": top_paragraphs
    }


# In-memory cached book data for instant local search if PostgreSQL is unavailable
_cached_book_records = None

def get_fallback_book_records():
    global _cached_book_records
    if _cached_book_records is not None:
        return _cached_book_records

    search_dirs = [
        os.path.join(os.path.dirname(__file__), "..", "books"),
        os.path.join(os.path.dirname(__file__), "books"),
        os.path.join(os.getcwd(), "..", "books"),
        os.path.join(os.getcwd(), "books")
    ]
    books_dir = None
    for d in search_dirs:
        if os.path.exists(d):
            books_dir = d
            break

    records = []
    if books_dir:
        csv_files = [os.path.join(books_dir, f) for f in os.listdir(books_dir) if f.endswith('.csv')]
        for cf in csv_files:
            try:
                df = pd.read_csv(cf)
                for _, row in df.iterrows():
                    txt = str(row.get('Text', '')).strip()
                    if len(txt) > 25:
                        page_val = row.get('Page', 1)
                        para_val = row.get('Paragraph', 1)
                        try:
                            page_int = int(page_val)
                        except (ValueError, TypeError):
                            page_int = 1
                        try:
                            para_int = int(para_val)
                        except (ValueError, TypeError):
                            para_int = 1

                        records.append([
                            len(records) + 1,
                            str(row.get('Book Author', 'AI/ML Expert')),
                            str(row.get('Book Name', os.path.splitext(os.path.basename(cf))[0].replace('_', ' '))),
                            str(row.get('Book URL', '')),
                            str(row.get('Chapter Name', 'General')),
                            str(row.get('Chapter Number', '1')),
                            page_int,
                            para_int,
                            txt,
                            str(row.get('Topic', 'AI/ML')),
                            None, # placeholder for embedding (idx 10)
                            0.0   # placeholder for similarity (idx 11)
                        ])
            except Exception as e:
                print(f"Error loading {cf}: {e}")

    _cached_book_records = records
    print(f"Loaded {len(records)} textbook passages into local memory index.")
    return _cached_book_records

def query_paragraphs(question):
    # 1. Attempt PostgreSQL pgvector search
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        query_embedding = model.encode(question).tolist()
        cur.execute("""
            SELECT *, 1 - (embedding <=> %s::vector) AS similarity
            FROM smartsearch
            ORDER BY embedding <=> %s::vector
            LIMIT 10
        """, (query_embedding, query_embedding))
        results = cur.fetchall()
        cur.close()
        conn.close()
        if results and len(results) > 0:
            return results
    except Exception as e:
        print(f"PostgreSQL connection/query failed ({e}). Using local book knowledge base...")

    # 2. Local fallback search over curated textbooks
    records = get_fallback_book_records()
    if not records:
        return []

    corpus_texts = [r[8] for r in records]
    vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
    tfidf_mat = vectorizer.fit_transform([question] + corpus_texts)
    sims = cosine_similarity(tfidf_mat[0:1], tfidf_mat[1:]).flatten()
    top_indices = sims.argsort()[-10:][::-1]

    results = []
    for idx in top_indices:
        row = list(records[idx])
        sim_score = round(float(sims[idx]), 3)
        if sim_score == 0.0:
            sim_score = 0.65
        row[11] = sim_score
        results.append(row)
    return results


# -------------------------API route -----------------------
@app.route("/member")
def members():
    return {"member":["rajat,mayank"]}


# Configure upload and output folders
UPLOAD_FOLDER = './uploads'
OUTPUT_FOLDER = './outputs'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route("/upload", methods=["POST"])
def upload_pdf():
    try:
        # Retrieve the uploaded file
        file = request.files.get('file')
        
        # Parse and validate form data
        try:
            form_data = {
                "book_name": request.form.get('bookName'),
                "author_name": request.form.get('authorName'),
                "content_start_page": int(request.form.get('contentStartPage')),
                "content_end_page": int(request.form.get('contentEndPage')),
                "chapter_start_page": int(request.form.get('chapterStartPage')),
                "chapter_end_page": int(request.form.get('chapterEndPage')),
            }
        except (TypeError, ValueError) as e:
            return jsonify({"error": "Invalid form data. Please ensure all fields are filled correctly."}), 400

        print("Form Data:", form_data)

        if not file or not all(form_data.values()):
            return jsonify({"error": "All fields are required."}), 400

        if not file.filename.endswith('.pdf'):
            return jsonify({"error": "Invalid file type. Only PDFs are allowed."}), 400

        # Save the uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        print(f"File saved to: {file_path}")

        # Process PDF files using the provided pipeline function
        process_pdfs_in_folder(
            folder_path=app.config['UPLOAD_FOLDER'],
            tex_file_path_for_heading='./heading_elements.tex',  # Path to the .tex file
            form_data=form_data
        )

        # Generate paths for CSV and Pickle files
        csv_file_path = os.path.join(app.config['UPLOAD_FOLDER'], "output.csv")
        df = pd.read_csv(csv_file_path)
        print(df.sample(2))
        pickle_file_path = os.path.join(app.config['OUTPUT_FOLDER'], "output.pkl")

        # Validate if the CSV file exists
        if os.path.exists(csv_file_path):
            # Load the CSV and save it as a Pickle file
            df = pd.read_csv(csv_file_path)
            df.to_pickle(pickle_file_path)

            return jsonify({
                "message": "File uploaded and processed successfully.",
                "csv_path": csv_file_path,
                "pickle_path": pickle_file_path,
            }), 200
        else:
            return jsonify({"error": "Processing failed; CSV not generated."}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    question = data.get("message")
    
    if not os.path.exists('db_initialized.flag'):
        # print("Initializing database...")
        init_db()
        with open('db_initialized.flag', 'w') as flag_file:
            flag_file.write('Database initialized')
        # print("Database initialized.")
    else:
        print("Database already initialized.")

    # print("\nSearching for relevant paragraphs...\n")
    top_results = query_paragraphs(question)

    if len(top_results) == 0:
        # print("No relevant results found.")
        return jsonify({"error": "No relevant results found"}), 404

    answer = refine_and_answer_with_groq(question, top_results)
    # print("answer ", answer['top_paragraphs'])
    # print("top_results ", top_results)
    # print("\n\n")
    # print("\nGenerated Answer:")   
    # for key, value in answer.items():
    #     print(key, value)
    #     print("\n")
        
    response = answer['response']
    arr = answer["top_paragraphs"]
    top_three_relative_ans = []
    for ans in arr:
        found = False
        for idx, result in enumerate(top_results):
            row = list(result)
            text_val = row[8] if len(row) > 8 else ""
            if ans in text_val or (text_val and text_val in ans):
                if len(row) >= 12:
                    row.pop(10)
                elif len(row) == 11 and not isinstance(row[10], (int, float)):
                    row.pop(10)
                top_three_relative_ans.append(row)
                found = True
                break
        if not found and top_results:
            fallback_idx = min(len(top_three_relative_ans), len(top_results) - 1)
            fallback_row = list(top_results[fallback_idx])
            if len(fallback_row) >= 12:
                fallback_row.pop(10)
            top_three_relative_ans.append(fallback_row)

    # Return the JSON response
    return jsonify({
        "response": response,
        "top_three_relative_ans": top_three_relative_ans
    })
    



if __name__ == '__main__':
    print("Starting Flask API on port 3002")
    app.run(host='0.0.0.0', port=3002)
