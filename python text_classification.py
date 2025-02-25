import requests
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Fetch data from API
url = "https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/"
params = {"size": 1000, "no_aggs": True}  # Fetch 1000 records for now
response = requests.get(url, params=params)

data = response.json()

# Convert to DataFrame
df = pd.json_normalize(data['hits']['hits'])

# Extract relevant columns
if 'fields.Product' in df.columns and 'fields.Consumer complaint narrative' in df.columns:
    df = df[['fields.Product', 'fields.Consumer complaint narrative']]
    df.columns = ['Product', 'Consumer complaint narrative']
else:
    raise ValueError("Required columns are missing in the fetched dataset.")

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Text preprocessing function
def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation + string.digits))
    tokens = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return ' '.join(tokens)

# Apply text preprocessing
df['cleaned_complaint'] = df['Consumer complaint narrative'].apply(preprocess_text)

# Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned_complaint'])

# Encode target labels
df['Product'] = df['Product'].astype('category').cat.codes
y = df['Product']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC()
}

# Train and evaluate models
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"--- {model_name} ---")
    print(classification_report(y_test, y_pred))

# Example new complaint
new_complaint = ["I am unable to get a response from the debt collector about my loan."]

# Preprocess and vectorize
new_complaint_cleaned = [preprocess_text(complaint) for complaint in new_complaint]
new_complaint_vectorized = vectorizer.transform(new_complaint_cleaned)

# Predict using the best model
best_model = models['Logistic Regression']  # Change if another model performs better
prediction = best_model.predict(new_complaint_vectorized)
predicted_category = df['Product'].cat.categories[prediction[0]]
print(f"Predicted Category: {predicted_category}")

