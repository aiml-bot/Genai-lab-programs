from transformers import pipeline

# Load the sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis")

def analyze_sentiment(text):
    """Analyze sentiment of the input text using Hugging Face pipeline."""
    result = sentiment_analyzer(text)
    label = result[0]['label']
    score = result[0]['score']
    return f"Sentiment: {label} (Confidence: {score:.2f})"

# Example usage
while True:
    user_input = input("Enter a sentence for sentiment analysis (or 'exit' to quit): ").strip()
    if user_input.lower() == 'exit':
        print("Exiting the program.")
        break
    print(analyze_sentiment(user_input))
