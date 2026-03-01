# CUSTOMER SUPPORT TICKET CLASSIFICATION SYSTEM

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
import warnings
warnings.filterwarnings("ignore")

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

nltk.download('stopwords')
nltk.download('wordnet')

# 2. Load Dataset
df = pd.read_csv("customer_support_tickets.csv")

print("\nDataset Loaded Successfully!")
print("Dataset Shape:", df.shape)
print("\nColumns:", df.columns)

# 3. Set Correct Columns
text_col_subject = "Ticket Subject"
text_col_description = "Ticket Description"
category_col = "Ticket Type"
priority_col = "Ticket Priority"

# Drop missing values
df = df.dropna(subset=[text_col_subject, text_col_description, category_col, priority_col])

# 4. Combine Subject + Description
df["combined_text"] = df[text_col_subject] + " " + df[text_col_description]

# 5. Text Cleaning
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

df["clean_text"] = df["combined_text"].apply(clean_text)

print("Text Cleaning Completed!")

# 6. TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["clean_text"])

# 7. Encode Labels
category_encoder = LabelEncoder()
priority_encoder = LabelEncoder()

y_category = category_encoder.fit_transform(df[category_col])
y_priority = priority_encoder.fit_transform(df[priority_col])

# 8. Train-Test Split
X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(
    X, y_category, test_size=0.2, random_state=42)

X_train_pri, X_test_pri, y_train_pri, y_test_pri = train_test_split(
    X, y_priority, test_size=0.2, random_state=42)

# 9. Train Models
model_category = LogisticRegression(max_iter=1000)
model_category.fit(X_train_cat, y_train_cat)

model_priority = LogisticRegression(max_iter=1000)
model_priority.fit(X_train_pri, y_train_pri)

print("\nModels Trained Successfully!")
# 10. Evaluation

print("\n================ CATEGORY MODEL ================")
y_pred_cat = model_category.predict(X_test_cat)
print("Accuracy:", accuracy_score(y_test_cat, y_pred_cat))
print(classification_report(y_test_cat, y_pred_cat))

print("\n================ PRIORITY MODEL ================")
y_pred_pri = model_priority.predict(X_test_pri)
print("Accuracy:", accuracy_score(y_test_pri, y_pred_pri))
print(classification_report(y_test_pri, y_pred_pri))

# 11. Visualization
# Priority Distribution
plt.figure()
sns.countplot(x=df[priority_col])
plt.title("Distribution of Ticket Priorities")
plt.xlabel("Priority Level")
plt.ylabel("Number of Tickets")
plt.show()

# Category vs Priority
plt.figure()
sns.countplot(x=category_col, hue=priority_col, data=df)
plt.xticks(rotation=45)
plt.title("Category vs Priority Distribution")
plt.show()

# Confusion Matrix - Priority
cm = confusion_matrix(y_test_pri, y_pred_pri)
plt.figure()
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix - Priority Model")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 12. Highest Priority Analysis
high_priority_counts = df[df[priority_col].str.lower() == "high"][category_col].value_counts()

print("\n===== Categories with Most HIGH Priority Tickets =====")
print(high_priority_counts)

# 13. Predict New Ticket
def predict_ticket(subject, description):
    text = subject + " " + description
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    
    category_pred = model_category.predict(vectorized)
    priority_pred = model_priority.predict(vectorized)
    
    category_name = category_encoder.inverse_transform(category_pred)
    priority_name = priority_encoder.inverse_transform(priority_pred)
    
    print("\nNew Ticket:")
    print("Subject:", subject)
    print("Description:", description)
    print("Predicted Category:", category_name[0])
    print("Predicted Priority:", priority_name[0])

# Example
predict_ticket(
    "Payment Failed",
    "Money was deducted but order not confirmed. Please resolve urgently."
)

# 14. Priority Analysis by Task Type
print("\n=========== PRIORITY ANALYSIS BY TASK TYPE ===========")

# Group by Ticket Type and Ticket Priority
priority_summary = df.groupby([category_col, priority_col]).size().unstack(fill_value=0)

print("\nPriority Count Table:\n")
print(priority_summary)

# High Priority Tasks
if "High" in priority_summary.columns:
    print("\nTasks with HIGH Priority Tickets:\n")
    high_sorted = priority_summary["High"].sort_values(ascending=False)
    print(high_sorted)

    print("\nTask with MOST HIGH priority tickets:")
    print(high_sorted.idxmax())

# Low Priority Tasks
if "Low" in priority_summary.columns:
    print("\nTasks with LOW Priority Tickets:\n")
    low_sorted = priority_summary["Low"].sort_values(ascending=False)
    print(low_sorted)

    print("\nTask with MOST LOW priority tickets:")
    print(low_sorted.idxmax())