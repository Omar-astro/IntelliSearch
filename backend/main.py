#region IMPORTS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, json, request, jsonify
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from backend.elmo_model import ELMoModel
from flask_cors import CORS
import pyterrier as pt
import pandas as pd
import random
import nltk
import json
import os
import re
#endregion

#region initialization
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

#Remove till next comment if you have no java problem
if 'JAVA_HOME' not in os.environ:
    java_paths = [
        r"C:\Program Files\Java\jdk-25.0.2",
        r"C:\Program Files\Java\jre-25.0.2",
    ]
    for path in java_paths:
        if os.path.exists(path):
            os.environ['JAVA_HOME'] = path
            break
#endregion

#----------------------------------------------------------

#region preprocessing

def preprocess2(text):
    if not isinstance(text, str):
        text = str(text)
    # Tokenization
    tokens = word_tokenize(text)

    text = text.lower()
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'\s+', ' ', text).strip()

    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    processed_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return ' '.join(processed_tokens)

def preprocess(text):
    if not isinstance(text, str):
        text = str(text)
    # Tokenization
    tokens = word_tokenize(text)

    text = text.lower()
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'\s+', ' ', text).strip()

    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    processed_tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]

    return ' '.join(processed_tokens)


def to_json_safe(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)

#endregion

#region LANGUAGE MODELS

if not pt.java.started():
    pt.java.init()

df = pd.read_csv('data/Articles.csv')
df2 = pd.read_csv('data/Preprocessed_Articles.csv')

# Text Searching
index_dir1 = os.path.join(os.getcwd(), "backend/indexing/index")
index_ref1 = pt.IndexFactory.of(index_dir1)
bm25_text = pt.BatchRetrieve(index_dir1, wmodel="BM25", num_results=10)
tfidf_text = pt.BatchRetrieve(index_dir1, wmodel="TF_IDF", num_results=10)
rm3_rtext = pt.rewrite.RM3(index_dir1, fb_terms=10, fb_docs=100)
rm3_text = bm25_text >> rm3_rtext

# Title Searching
index_dir2 = os.path.join(os.getcwd(), "backend/indexing/indexTitle")
index_ref2 = pt.IndexFactory.of(index_dir2)
bm25_title = pt.BatchRetrieve(index_dir2, wmodel="BM25", num_results=10)
tfidf_title = pt.BatchRetrieve(index_dir2, wmodel="TF_IDF", num_results=10)
rm3_rtitle = pt.rewrite.RM3(index_dir2, fb_terms=10, fb_docs=100)
rm3_title = bm25_title >> rm3_rtitle

# SubTitle Searching
index_dir3 = os.path.join(os.getcwd(), "backend/indexing/indexSubTitle")
index_ref3 = pt.IndexFactory.of(index_dir3)
bm25_subtitle = pt.BatchRetrieve(index_dir3, wmodel="BM25", num_results=10)
tfidf_subtitle = pt.BatchRetrieve(index_dir3, wmodel="TF_IDF", num_results=10)
rm3_rsubtitle = pt.rewrite.RM3(index_dir3, fb_terms=10, fb_docs=100)
rm3_subtitle = bm25_subtitle >> rm3_rsubtitle

elmo_model = ELMoModel()

#endregion

#----------------------------------------------------------
#region COSINE SIMILARITY

def search_cosine(column_name, query, top_k=10):
    df2[column_name] = df2[column_name].fillna('')

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df2[column_name])

    query_proc = preprocess2(query)
    query_vector = vectorizer.transform([query_proc])

    scores = cosine_similarity(query_vector, tfidf_matrix).flatten()

    results_df = df.copy()
    results_df['score'] = scores

    results = results_df[results_df['score'] > 0].copy()
    results = results.sort_values('score', ascending=False)
    top_results = results.head(top_k)

    top_results = top_results.copy()
    top_results['docno'] = top_results.index.astype(str)

    list5 = {
        "score": top_results['score'].tolist(),
        "docno": top_results['docno'].tolist()
    }
    return list5

#endregion
#----------------------------------------------------------
#region N-GRAM MODELS
from collections import Counter, defaultdict

def prep_documents(column_name):
    # Ensure no nans
    df2[column_name] = df2[column_name].fillna('')
    documents = {}
    for i, text in enumerate(df2[column_name]):
        documents[i] = str(text).split()
    return documents


# =============================================
# UNIGRAM MODEL
# =============================================
def build_unigram_model(tokenized_docs):
    unigram_models = {}
    for doc_id, tokens in tokenized_docs.items():
        total_terms = len(tokens)
        if total_terms == 0:
            unigram_models[doc_id] = {}
            continue
            
        term_freqs = Counter(tokens)
        unigram_models[doc_id] = {
            term: freq / total_terms
            for term, freq in term_freqs.items()
        }
    return unigram_models

def search_unigram(query, top_k=10, tokenized_documents=None):
    query_proc = preprocess2(query)
    tokenized_query = query_proc.split()
    
    unigram_models = build_unigram_model(tokenized_documents)
    
    query_probs = {}
    for doc_id, model in unigram_models.items():
        if not tokenized_query:
            query_probs[doc_id] = 0
            continue
            
        prob = 1.0
        for term in tokenized_query:
            term_prob = model.get(term, 0)
            prob *= term_prob
        query_probs[doc_id] = prob

    sorted_results = sorted(query_probs.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    # map back to nice outputs
    docnos = []
    scores = []
    for doc_id, score in sorted_results:
        docnos.append(str(doc_id))
        scores.append(score)
        
    return {"docno": docnos, "score": scores}
# tokenized_documents = prep_documents("Text")
# print("=== UNIGRAM RESULTS ===")
# print(search_unigram("pandas"))

# =============================================
# BIGRAM MODEL (With Laplace Smoothing)
# =============================================
def compute_bigram_query_prob(doc_tokens, query_tokens):
    if not query_tokens or not doc_tokens:
        return 0.0
        
    unigram_freq = defaultdict(int)
    bigram_freq = defaultdict(int)

    for i in range(len(doc_tokens)):
        unigram_freq[doc_tokens[i]] += 1
        if i < len(doc_tokens) - 1:
            bigram = (doc_tokens[i], doc_tokens[i + 1])
            bigram_freq[bigram] += 1

    query_bigrams = list(zip(query_tokens[:-1], query_tokens[1:]))
    total_tokens = len(doc_tokens)
    vocab_size = len(unigram_freq)

    first_word = query_tokens[0]
    p_query = (unigram_freq[first_word] + 1) / (total_tokens + vocab_size) if total_tokens > 0 else 0

    for (w1, w2) in query_bigrams:
        # P(w2|w1) = (Count(w1, w2) + 1) / (Count(w1) + Vocab Size)
        p_query *= (bigram_freq[(w1, w2)] + 1) / (unigram_freq[w1] + vocab_size)

    return p_query  # Removed round(..., 6) as scores get naturally tiny

def search_bigram(query, top_k=10, tokenized_documents=None):
    query_proc = preprocess2(query)
    tokenized_query = query_proc.split()
    
    query_probs = {}
    for doc_id, tokens in tokenized_documents.items():
        prob = compute_bigram_query_prob(tokens, tokenized_query)
        query_probs[doc_id] = prob
        
    sorted_results = sorted(query_probs.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    docnos = []
    scores = []
    for doc_id, score in sorted_results:
        docnos.append(str(doc_id))
        scores.append(score)
        
    return {"docno": docnos, "score": scores}
# print("=== BIGRAM RESULTS ===")
# print(search_bigram("pandas database"))

# =============================================
# TRIGRAM MODEL (With Laplace Smoothing)
# =============================================
def compute_trigram_query_prob(doc_tokens, query_tokens):
    if len(query_tokens) < 3 or not doc_tokens:
        return 0.0

    query_trigrams = list(zip(
        query_tokens[:-2],
        query_tokens[1:-1],
        query_tokens[2:]
    ))

    unigram_freq = defaultdict(int)
    bigram_freq = defaultdict(int)
    trigram_freq = defaultdict(int)

    for i in range(len(doc_tokens)):
        unigram_freq[doc_tokens[i]] += 1
        
        if i < len(doc_tokens) - 1:
            bigram = (doc_tokens[i], doc_tokens[i + 1])
            bigram_freq[bigram] += 1

        if i < len(doc_tokens) - 2:
            trigram = (doc_tokens[i], doc_tokens[i + 1], doc_tokens[i + 2])
            trigram_freq[trigram] += 1

    total_tokens = len(doc_tokens)
    vocab_size = len(unigram_freq)  # For Laplace smoothing

    # Initialize probability to 1.0, not zero
    p_query = 1.0

    for (w1, w2, w3) in query_trigrams:
        bigram = (w1, w2)
        # Laplace (+1) smoothing for trigrams:
        # P(w3|w1, w2) = (Count(w1, w2, w3) + 1) / (Count(w1, w2) + Vocab Size)
        p_query *= (trigram_freq[(w1, w2, w3)] + 1) / (bigram_freq[bigram] + vocab_size)

    return p_query  # Removed round(..., 6) because decimals get very small down the sequence

def search_trigram(query, top_k=10, tokenized_documents=None):
    query_proc = preprocess2(query)
    tokenized_query = query_proc.split()
    
    query_probs = {}
    for doc_id, tokens in tokenized_documents.items():
        prob = compute_trigram_query_prob(tokens, tokenized_query)
        query_probs[doc_id] = prob
        
    sorted_results = sorted(query_probs.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    docnos = []
    scores = []
    for doc_id, score in sorted_results:
        docnos.append(str(doc_id))
        scores.append(score)
        
    return {"docno": docnos, "score": scores}
# print("=== TRIGRAM RESULTS ===")
# print(search_trigram("pandas open source database"))

#endregion
#----------------------------------------------------------

#region AUTOCOMPLETE

word_cont = {}

def process_corpus(corpus, weight):
    for sentence in corpus:
        if not isinstance(sentence, str):
            continue
        tokens = word_tokenize(sentence)
        for i in range(len(tokens)-1):
            word = tokens[i]
            word_after = tokens[i+1]
            
            # Check both the word and the word immediately after
            if re.match(r'^\w+$', word) and re.match(r'^\w+$', word_after):
                if word not in word_cont:
                    word_cont[word] = {}
                if word_after not in word_cont[word]:
                    word_cont[word][word_after] = 0
                word_cont[word][word_after] += weight

process_corpus(df2['Text'], 1)
if 'Title' in df2.columns:
    process_corpus(df2['Title'], 3)
if 'SubTitle' in df2.columns:
    process_corpus(df2['SubTitle'], 2)

#endregion

#================================================
#=================== FLASK ======================
#================================================

app = Flask(__name__)

CORS(app)

@app.route('/result', methods=['POST'])
def query_search_result():
    data = request.get_json() # {'query': 'What is the capital of France?'}

    query = data.get('query', '')
    algo = data.get('algorithm', '')
    field = data.get('field', '')

    query = preprocess(query)

    try:
        with open("data/recommend.json", "r") as file:
            Mjson = json.load(file)
    except FileNotFoundError:
        Mjson = {}
    
    if query not in Mjson:
        Mjson[query] = 1
    else:
        Mjson[query] += 1

    with open("data/recommend.json", "w") as file:
        json.dump(Mjson, file, indent=4)

    if algo == 'bm25':
        if field == 'text':
            results = bm25_text.search(query)
        elif field == 'title':
            results = bm25_title.search(query)
        elif field == 'subtitle':
            results = bm25_subtitle.search(query)
        # print("==========used bm25==========")
    elif algo == 'tfidf':
        if field == 'text':
            results = tfidf_text.search(query)
        elif field == 'title':
            results = tfidf_title.search(query)
        elif field == 'subtitle':
            results = tfidf_subtitle.search(query)
        # print("==========used tfidf==========")
    elif algo == 'cosine':
        if field == 'text':
            results = search_cosine("Text", query)
        elif field == 'title':
            results = search_cosine("Title", query)
        elif field == 'subtitle':
            results = search_cosine("SubTitle", query)
        # print("==========used cosine==========")
    elif algo == 'rm3':
        if field == 'text':
            expanded_query = rm3_text.search(query).iloc[0]["query"]
            expanded_query_formatted = ' '.join(expanded_query.split()[1:])
            results = bm25_text.search(expanded_query_formatted)
        elif field == 'title':
            expanded_query = rm3_title.search(query).iloc[0]["query"]
            expanded_query_formatted = ' '.join(expanded_query.split()[1:])
            results = bm25_title.search(expanded_query_formatted)
        elif field == 'subtitle':
            expanded_query = rm3_subtitle.search(query).iloc[0]["query"]
            expanded_query_formatted = ' '.join(expanded_query.split()[1:])
            results = bm25_subtitle.search(expanded_query_formatted)
        # print("==========used rm3==========")
    elif algo == 'gram':
        if field == 'text':
            tokenized_documents = prep_documents("Text")
        elif field == 'title':
            tokenized_documents = prep_documents("Title")
        elif field == 'subtitle':
            tokenized_documents = prep_documents("SubTitle")
        
        if len(query.split()) == 1:
            results = search_unigram(query, tokenized_documents=tokenized_documents)
        elif len(query.split()) == 2:
            results = search_bigram(query, tokenized_documents=tokenized_documents)
        elif len(query.split()) >= 3:
            results = search_trigram(query, tokenized_documents=tokenized_documents)
        # print("==========used unigram==========")
    elif algo == 'elmo':
        results = elmo_model.search(query)
        # print("==========used tfidf==========")

    dic = {}
    for i in range(len(results)):
        doc_idx = int(results['docno'][i])
        full_text = df['Text'][doc_idx]
        text_sample = full_text[:300] + "..." if len(full_text) > 150 else full_text
        raw_score = to_json_safe(results['score'][i])
        score_value = round(float(raw_score), 2) if isinstance(raw_score, (int, float)) else 0.0
        
        dic["doc " + str(i)] = {
            "id": int(doc_idx),
            "score": score_value,
            "title": to_json_safe(df['Title'][doc_idx]),
            "SubTitle": to_json_safe(df['SubTitle'][doc_idx]),
            "Author": to_json_safe(df['Author'][doc_idx]),
            "date": to_json_safe(df['date'][doc_idx]),
            "text": to_json_safe(text_sample),
            "image": to_json_safe(df['image'][doc_idx]),
            "link": to_json_safe(df['Link'][doc_idx])
        }

    return jsonify(dic)

@app.route('/suggest', methods=['POST'])
def get_suggestions():
    data = request.get_json()
    old_query = data.get('query', '')
    
    if not old_query or not old_query.strip():
        return jsonify([])
        
    query = preprocess2(old_query)
    query_words = query.split()
    
    if not query_words:
        return jsonify([old_query])
        
    last_word = query_words[-1]
    
    list2 = []
    for word in word_cont.keys():
        if isinstance(word, str) and word.startswith(last_word):
            list2.append(word)
    
    list1 = []

    if len(list2) > 1:
        old_words = old_query.split()
        prefix = " ".join(old_words[:-1]) + " " if len(old_words) > 1 else ""
        list1 = [prefix + word for word in list2]
        return jsonify(list1)
    
    try:
        word1 = last_word
        
        if word1 not in word_cont:
            stemmed_query = preprocess(old_query)
            stemmed_words = stemmed_query.split()
            if stemmed_words:
                word1 = stemmed_words[-1]
        
        if word1 in word_cont:
            sorted_items = sorted(word_cont[word1].items(), key=lambda item: item[1], reverse=True)
            limit = min(5, len(sorted_items))
            top_words = [word for word, count in sorted_items[:limit]]

            for word in top_words:
                list1.append(old_query + " " + word)
                
        if not list1:
            list1.append(old_query)
            
    except Exception:
        if not list1:
            list1.append(old_query)
    
    return jsonify(list1)

@app.route('/Feedback', methods=['POST'])
def submit_feedback():
    data = request.get_json()
    query = data.get('query', '')
    docid = data.get('docid', '')
    model = data.get('model', '')
    relevance: bool = data.get('relevant', False)
    retrived = data.get('retrived', [])
    query = preprocess2(query)

    # region setting_json

    with open("data/Feedback.json", "r") as file:
        Mjson = json.load(file)

    Mjson.setdefault("relevant", {})
    Mjson["relevant"].setdefault(str(query), {})
    Mjson["relevant"][str(query)].setdefault(str(docid), {"relevant_count": 0, "non_relevant_count": 0})

    if relevance:
        Mjson["relevant"][str(query)][str(docid)]["relevant_count"] = Mjson["relevant"][str(query)][str(docid)].get("relevant_count", 0) + 1
    else:
        Mjson["relevant"][str(query)][str(docid)]["non_relevant_count"] = Mjson["relevant"][str(query)][str(docid)].get("non_relevant_count", 0) + 1

    Mjson.setdefault(str(model), {})
    Mjson[str(model)].setdefault(str(query), {})
    Mjson[str(model)][str(query)].setdefault("retrieved", [])
    for item in retrived:
        if item not in Mjson[str(model)][str(query)]["retrieved"]:
            Mjson[str(model)][str(query)]["retrieved"].append(item)
            
    #all relevant for query
    t1 = [x for x in Mjson['relevant'][str(query)].keys() if Mjson['relevant'][str(query)][str(x)]['relevant_count'] >= Mjson['relevant'][str(query)][str(x)]['non_relevant_count']]
    #all relevant from retrived (cast to string for correct comparison)
    t2 = [x for x in retrived if str(x) in t1]
    persision = len(t2) / len(retrived) if len(retrived) > 0 else 0
    recall = len(t2) / len(t1) if len(t1) > 0 else 0
    Mjson[str(model)][str(query)].setdefault("precision", 0)
    Mjson[str(model)][str(query)]["precision"] = persision
    Mjson[str(model)][str(query)].setdefault("recall", 0)
    Mjson[str(model)][str(query)]["recall"] = recall
    Mjson[str(model)][str(query)].setdefault("F1score", 0)
    Mjson[str(model)][str(query)]["F1score"] = 2 * (persision * recall) / (persision + recall) if (persision + recall) > 0 else 0
    valid_qs = [q for q in Mjson[str(model)].keys() if isinstance(Mjson[str(model)][q], dict) and "F1score" in Mjson[str(model)][q]]
    Mjson[str(model)]["average_score"] = sum([Mjson[str(model)][q]["F1score"] for q in valid_qs]) / len(valid_qs) if valid_qs else 0

    with open("data/Feedback.json", "w") as file:
        json.dump(Mjson, file, indent=4)

    #endregion

    # Build the dictionary of model scores mapping exactly to frontend IDs
    scores = {}
    for model_key in Mjson.keys():
        if model_key == "relevant":
            continue
        scores[model_key] = Mjson[model_key].get('average_score', 0)

    return jsonify(scores)

@app.route('/updatescore', methods=['POST'])
def update_score():
    with open("data/Feedback.json", "r") as file:
        Mjson = json.load(file)
    
    scores = {}
    for model_key in Mjson.keys():
        if model_key == "relevant":
            continue
        scores[model_key] = Mjson[model_key].get('average_score', 0)

    return jsonify(scores)

@app.route('/recommend', methods=['POST'])
def recommend_models():
    try:
        with open("data/recommend.json", "r") as file:
            Mjson = json.load(file)
    except FileNotFoundError:
        Mjson = {}

    dic = {}
    used_docs = set()

    RANDOM_DOCS = 4
    SEARCH_DOCS = 6

    queries = list(Mjson.keys())
    current_doc = 0

    if len(queries) > 0:
        total_weight = sum(Mjson.values())

        for query in queries:
            amount = round(
                SEARCH_DOCS * (Mjson[query] / total_weight)
            )

            amount = max(1, amount)
            results = bm25_text.search(query)
            top_docs = []
            limit = min(10, len(results))

            for i in range(limit):
                doc_idx = int(results['docno'][i])
                top_docs.append(doc_idx)

            random.shuffle(top_docs)
            picked = 0

            for doc_idx in top_docs:
                if picked >= amount:
                    break
                if doc_idx in used_docs:
                    continue

                used_docs.add(doc_idx)
                full_text = df['Text'][doc_idx]
                text_sample = (
                    full_text[:300] + "..."
                    if len(full_text) > 300
                    else full_text
                )
                dic[f"doc {current_doc}"] = {
                    "id": int(doc_idx),
                    "title": to_json_safe(df['Title'][doc_idx]),
                    "SubTitle": to_json_safe(df['SubTitle'][doc_idx]),
                    "Author": to_json_safe(df['Author'][doc_idx]),
                    "date": to_json_safe(df['date'][doc_idx]),
                    "text": to_json_safe(text_sample),
                    "image": to_json_safe(df['image'][doc_idx]),
                    "link": to_json_safe(df['Link'][doc_idx])
                }

                current_doc += 1
                picked += 1

    random_indices = list(df.index)
    random.shuffle(random_indices)
    added_random = 0

    for doc_idx in random_indices:
        if added_random >= RANDOM_DOCS:
            break
        if doc_idx in used_docs:
            continue
        used_docs.add(doc_idx)
        full_text = df['Text'][doc_idx]
        text_sample = (
            full_text[:300] + "..."
            if len(full_text) > 300
            else full_text
        )
        dic[f"doc {current_doc}"] = {
            "id": int(doc_idx),
            "title": to_json_safe(df['Title'][doc_idx]),
            "SubTitle": to_json_safe(df['SubTitle'][doc_idx]),
            "Author": to_json_safe(df['Author'][doc_idx]),
            "date": to_json_safe(df['date'][doc_idx]),
            "text": to_json_safe(text_sample),
            "image": to_json_safe(df['image'][doc_idx]),
            "link": to_json_safe(df['Link'][doc_idx])
        }
        current_doc += 1
        added_random += 1

    return jsonify(dic)


if __name__ == '__main__':
    app.run(debug=False, port=5000)