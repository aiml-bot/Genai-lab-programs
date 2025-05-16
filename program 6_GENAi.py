# Step 1: Install required libraries (uncomment and run this cell first)
!pip install transformers ipywidgets torch

# Step 2: Import the sentiment analysis pipeline
from transformers import pipeline

# Step 3: Load the sentiment analysis model
print("Loading Sentiment Analysis Model...")
sentiment_analyzer = pipeline("sentiment-analysis")
print("Model Loaded Successfully.\n")

# Step 4: Define the sentiment analysis function
def analyze_sentiment(text):
    """
    Analyze the sentiment of a given text input.

    Args:
        text (str): Input sentence or paragraph.

    Returns:
        dict: Sentiment label and confidence score.
    """
    result = sentiment_analyzer(text)[0]
    label = result['label']  # Sentiment Label (POSITIVE/NEGATIVE)
    score = result['score']  # Confidence Score

    print(f"\nInput Text: {text}")
    print(f"Sentiment: {label} (Confidence: {score:.4f})\n")

    return result

# Step 5: Example real-world application: Customer feedback analysis
customer_reviews = [
    "The product is amazing! I love it so much.",
    "I'm very disappointed. The service was terrible.",
    "It was an average experience, nothing special.",
    "Absolutely fantastic quality! Highly recommended.",
    "Not great, but not the worst either."
]

# Step 6: Analyze sentiment for each review
print("\nCustomer Sentiment Analysis Results:")
for review in customer_reviews:
    analyze_sentiment(review)
