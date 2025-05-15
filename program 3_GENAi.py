PROGRAM -3 
Train a custom Word2Vec model on a small dataset. Train embeddings on a 
domain-specific corpus (e.g., legal, medical) and analyze how embeddings 
capture domain-specific semantics. 
 
import gensim 
from gensim.models import Word2Vec 
import nltk 
from nltk.tokenize import word_tokenize 
import matplotlib.pyplot as plt 
from sklearn.manifold import TSNE 
 
# Medical corpus 
medical_corpus = [ 
    "The doctor diagnosed the patient with diabetes.", 
    "Insulin is used to treat diabetes.", 
    "A cardiologist specializes in heart diseases.", 
    "Patients with hypertension should reduce salt intake.", 
    "Antibiotics treat bacterial infections.", 
    "Vaccines help build immunity.", 
    "Surgery removes tumors.", 
    "A neurologist treats nervous system disorders.", 
] 
 
# Tokenization 
nltk.download('punkt') 
tokenized_corpus = [word_tokenize(sentence.lower()) for sentence in 
medical_corpus] 
 
# Train Word2Vec model 
model = Word2Vec(sentences=tokenized_corpus, vector_size=50, window=5, 
min_count=1, workers=4) 
 
# Save model 
model.save('medical_word2vec.model') 
 
def plot_embeddings(model): 
    words = list(model.wv.index_to_key) 
    word_vectors = model.wv[words] 
     
    # Reduce dimensions using t-SNE 
    tsne = TSNE(n_components=2, random_state=42) 
    reduced_vectors = tsne.fit_transform(word_vectors) 
     
    plt.figure(figsize=(8, 5)) 
    for i, word in enumerate(words): 
        x, y = reduced_vectors[i] 
        plt.scatter(x, y) 
        plt.annotate(word, (x, y), fontsize=10) 
     
    plt.title("Word Embeddings Visualization") 
    plt.show() 
 
# Plot word embeddings 
plot_embeddings(model) 
 
# Find similar words 
def find_similar_words(word): 
    if word in model.wv: 
        similar_words = model.wv.most_similar(word, topn=5) 
        print(f"Words similar to '{word}':") 
        for similar, score in similar_words: 
            print(f"{similar} (Similarity: {score:.2f})") 
    else: 
        print(f"'{word}' not found in vocabulary") 
 
# Example usage 
find_similar_words("diabetes") 
 
 
