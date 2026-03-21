import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# Setup paths using absolute directory references
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'dataset')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

os.makedirs(MODELS_DIR, exist_ok=True)

def train_diabetes():
    print("Training Diabetes Model...")
    df = pd.read_csv(os.path.join(DATA_DIR, 'diabetes.csv'))
    X = df.drop(columns=['Outcome'])
    y = df['Outcome']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    models = {
        'Logistic Regression': LogisticRegression(),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(probability=True)
    }
    
    best_model = None
    best_acc = 0
    for name, model in models.items():
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        print(f"  {name} Accuracy: {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_model = model
            
    print(f"  -> Best Model: {best_model.__class__.__name__} with accuracy {best_acc:.4f}\n")
    joblib.dump(best_model, os.path.join(MODELS_DIR, 'diabetes_model.pkl'))
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'diabetes_scaler.pkl'))

def train_heart():
    print("Training Heart Disease Model...")
    df = pd.read_csv(os.path.join(DATA_DIR, 'heart.csv'))
    
    # Manual encoding
    df['Sex'] = df['Sex'].map({'M': 1, 'F': 0})
    df['ChestPainType'] = df['ChestPainType'].map({'ATA': 0, 'NAP': 1, 'ASY': 2, 'TA': 3})
    df['RestingECG'] = df['RestingECG'].map({'Normal': 0, 'ST': 1, 'LVH': 2})
    df['ExerciseAngina'] = df['ExerciseAngina'].map({'N': 0, 'Y': 1})
    df['ST_Slope'] = df['ST_Slope'].map({'Up': 0, 'Flat': 1, 'Down': 2})
    
    X = df.drop(columns=['HeartDisease'])
    y = df['HeartDisease']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    models = {
        'Logistic Regression': LogisticRegression(),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(probability=True)
    }
    
    best_model = None
    best_acc = 0
    for name, model in models.items():
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        print(f"  {name} Accuracy: {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_model = model
            
    print(f"  -> Best Model: {best_model.__class__.__name__} with accuracy {best_acc:.4f}\n")
    joblib.dump(best_model, os.path.join(MODELS_DIR, 'heart_model.pkl'))
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'heart_scaler.pkl'))

def train_parkinsons():
    print("Training Parkinson's Disease Model...")
    df = pd.read_csv(os.path.join(DATA_DIR, 'parkinsons.csv'))
    
    # Drop the name column as it is an ID
    X = df.drop(columns=['name', 'status'])
    y = df['status']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    models = {
        'Logistic Regression': LogisticRegression(),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(probability=True)
    }
    
    best_model = None
    best_acc = 0
    for name, model in models.items():
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        print(f"  {name} Accuracy: {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_model = model
            
    print(f"  -> Best Model: {best_model.__class__.__name__} with accuracy {best_acc:.4f}\n")
    joblib.dump(best_model, os.path.join(MODELS_DIR, 'parkinsons_model.pkl'))
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'parkinsons_scaler.pkl'))

if __name__ == '__main__':
    print("========================================")
    print("Starting Model Training Pipeline...")
    print("========================================\n")
    train_diabetes()
    train_heart()
    train_parkinsons()
    print("All models trained and saved to the 'models' directory.")
