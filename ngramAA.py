import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import time

# Function to load and preprocess the data from Hugging Face datasets
def load_data():
    # Load the Reddit dataset
    dataset = load_dataset("reddit", split="train[:100000]", trust_remote_code=True)  # Limiting to 100k samples for this example

    # Convert to pandas DataFrame for easier preprocessing
    df = pd.DataFrame(dataset)
    # Use 'author' as our target for authorship attribution
    # Keep only authors with at least 8 comments
    author_counts = df['author'].value_counts()
    authors_to_keep = author_counts[author_counts >= 8].index
    data = df[df['author'].isin(authors_to_keep)]
    
    # Extract text and author information
    texts = data['content']
    authors = data['author']
    
    return texts, authors

def create_pipeline(ngram_type='word'):
    if ngram_type == 'word':
        vectorizer = CountVectorizer(ngram_range=(1, 3), max_features=10000)
    else:  # character n-grams
        vectorizer = CountVectorizer(ngram_range=(1, 5), analyzer='char', max_features=10000)
    
    return Pipeline([
        ('vectorizer', vectorizer),
        ('classifier', SVC(kernel='linear', C=1.0, random_state=42))
    ])

def train_and_evaluate(X_train, X_test, y_train, y_test, pipeline, ngram_type, label_encoder):
    start_time = time.time()
    pipeline.fit(X_train, y_train)
    train_time = time.time() - start_time

    start_time = time.time()
    y_pred = pipeline.predict(X_test)
    predict_time = time.time() - start_time

    print(f"\n{ngram_type.capitalize()} N-gram Results:")
    print(f"Training time: {train_time:.2f} seconds")
    print(f"Prediction time: {predict_time:.2f} seconds")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    report = classification_report(y_test, y_pred, output_dict=True)
    
    return report['accuracy'], train_time, predict_time

def main():
    print("Loading data...")
    texts, authors = load_data()

    print("Encoding author labels...")
    label_encoder = LabelEncoder()
    authors_encoded = label_encoder.fit_transform(authors)

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        texts, authors_encoded, test_size=0.2, random_state=42, stratify=authors_encoded
    )

    # Word-level n-grams
    word_pipeline = create_pipeline('word')
    word_accuracy, word_train_time, word_predict_time = train_and_evaluate(
        X_train, X_test, y_train, y_test, word_pipeline, 'word', label_encoder
    )

    # Character-level n-grams
    char_pipeline = create_pipeline('char')
    char_accuracy, char_train_time, char_predict_time = train_and_evaluate(
        X_train, X_test, y_train, y_test, char_pipeline, 'character', label_encoder
    )

    print("\nOverall Comparison:")
    print(f"Word-level N-grams - Training time: {word_train_time:.2f}s, Prediction time: {word_predict_time:.2f}s")
    print(f"Char-level N-grams - Training time: {char_train_time:.2f}s, Prediction time: {char_predict_time:.2f}s")

    print("\nComparison:")
    print(f"Word-level N-grams - Accuracy: {word_accuracy:.4f}, Training time: {word_train_time:.2f}s, Prediction time: {word_predict_time:.2f}s")
    print(f"Char-level N-grams - Accuracy: {char_accuracy:.4f}, Training time: {char_train_time:.2f}s, Prediction time: {char_predict_time:.2f}s")

    if word_accuracy > char_accuracy:
        print("\nWord-level n-grams performed better in terms of accuracy.")
    elif char_accuracy > word_accuracy:
        print("\nCharacter-level n-grams performed better in terms of accuracy.")
    else:
        print("\nBoth methods performed equally in terms of accuracy.")

    # Example prediction
    sample_text = "And that is, hands down, the coolest aspect of the game. It rewards creativity, careful planning, and experimenting with unconventional ideas. There is no optimum build or gear allocation. You can do things that the designers never even dreamed of and come up with a very effective build that might be unlike anything that anyone else does."
    word_author = word_pipeline.predict([sample_text])
    char_author = char_pipeline.predict([sample_text])
    print(f"\nSample text: '{sample_text}'")
    print(f"Word n-gram prediction: {label_encoder.inverse_transform(word_author)[0]}")
    print(f"Char n-gram prediction: {label_encoder.inverse_transform(char_author)[0]}")

if __name__ == "__main__":
    main()