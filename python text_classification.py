import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.combine import SMOTETomek
from sklearn.preprocessing import LabelEncoder

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Load dataset
df = pd.read_csv("https://files.consumerfinance.gov/ccdb/complaints.csv.zip", compression='zip', low_memory=False, nrows=500000)
df = df[['Product', 'Issue', 'Sub-issue']].dropna()

# Define categories
category_mapping = {
    "Credit reporting or other personal consumer reports": 0,
    "Debt collection": 1,
    "Payday loan, title loan, personal loan, or advance loan": 2,
    "Mortgage": 3
}
df['Product'] = df['Product'].map(category_mapping)
df.dropna(inplace=True)

# Combine Issue and Sub-issue columns as input text
df['text'] = df['Issue'] + ' ' + df['Sub-issue']

# Visualize class distribution before balancing
sns.countplot(x=df['Product'])
plt.title("Class Distribution Before Balancing")
plt.show()

# Text vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['text']).toarray()
y = df['Product'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle class imbalance
smote_tomek = SMOTETomek()
X_train_resampled, y_train_resampled = smote_tomek.fit_resample(X_train, y_train)

# Visualize class distribution after balancing
sns.countplot(x=y_train_resampled)
plt.title("Class Distribution After Balancing")
plt.show()

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_resampled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_resampled, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define BiLSTM model
class BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
    
    def forward(self, x):
        x = x.unsqueeze(1)
        _, (hn, _) = self.lstm(x)
        out = self.fc(torch.cat((hn[0], hn[1]), dim=1))
        return out

# Model parameters
input_dim = X_train_tensor.shape[1]
hidden_dim = 128
output_dim = len(category_mapping)
model = BiLSTM(input_dim, hidden_dim, output_dim)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
epochs = 2
for epoch in range(epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Evaluate the model
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")
