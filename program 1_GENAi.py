PROGRAM 1: 
 
1. Explore pre-trained word vectors. Explore word relationships using vector arithmetic. Perform 
arithmetic operations and analyze results. 
 
import gensim.downloader as api 
 
# Load pre-trained Word2Vec model (Google News) 
print("Loading model... (This may take a while)") 
model = api.load("word2vec-google-news-300") 
print("Model loaded!") 
 
# Function to find similar words 
def find_similar(word): 
    try: 
        similar_words = model.most_similar(word) 
        print(f"\nWords similar to '{word}':") 
        for w, score in similar_words[:5]:  # Show top 5 
            print(f"{w}: {score:.4f}") 
    except KeyError: 
        print(f"'{word}' not found in the vocabulary.") 
 
# Function to perform word arithmetic 
def word_arithmetic(word1, word2, word3): 
    try: 
        result = model.most_similar(positive=[word1, word2], negative=[word3]) 
        print(f"\n'{word1}' - '{word3}' + '{word2}' = '{result[0][0]}' (Most similar 
word)") 
    except KeyError as e: 
        print(f"Error: {e}") 
 
# Function to check similarity between two words 
def check_similarity(word1, word2): 
    try: 
        similarity = model.similarity(word1, word2) 
        print(f"\nSimilarity between '{word1}' and '{word2}': {similarity:.4f}") 
    except KeyError as e: 
        print(f"Error: {e}") 
 
# Function to find the odd one out 
def odd_one_out(words): 
    try: 
        odd = model.doesnt_match(words) 
        print(f"\nOdd one out from {words}: {odd}") 
    except KeyError as e: 
        print(f"Error: {e}") 
 
# Run the functions 
find_similar("king") 
word_arithmetic("king", "woman", "man")  # Expected output: "queen" 
check_similarity("king", "queen") 
odd_one_out(["apple", "banana", "grape", "car"])  # "car" should be the odd one