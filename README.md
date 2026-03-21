# Multi-Disease Prediction System with MLOps Pipeline

A machine learning web application that predicts **Diabetes**, **Heart Disease**, and **Parkinson's Disease** from patient health parameters. This project features an end-to-end MLOps pipeline including a Streamlit UI, Docker containerization, and GitHub Actions CI.

## 🚀 Execution Guide (How to run locally)

### 1. Prerequisites
- Python 3.9+ 
- Docker (optional, for containerization)

### 2. Setup
Clone the repository, ensure your datasets are inside the `dataset/` folder (`diabetes.csv`, `heart.csv`, `parkinsons.csv`), and install dependencies:
```bash
pip install -r requirements.txt
```

### 3. Model Training
Run the training script to generate the models and scalers into the `models/` directory:
```bash
python src/train.py
```

### 4. Running the Streamlit App
Start the web application:
```bash
streamlit run app/app.py
```
The app will be available at `http://localhost:8501`.

### 5. Running with Docker
```bash
docker build -t disease-mlops .
docker run -p 8501:8501 disease-mlops
```

---

## 🎯 Viva Questions and Answers

**Q1: Why did you use multiple models instead of just one?**
**A1:** Different diseases have different feature characteristics. We trained multiple models (Logistic Regression, Random Forest, SVM) for each disease and compared them based on accuracy. We then selected and saved the best performing model dynamically during the training phase.

**Q2: What role does MLOps play in this project?**
**A2:** MLOps automates the pipeline. Our GitHub Actions file (`.github/workflows/main.yml`) ensures Continuous Integration (CI) by automatically linting the code, training models to verify data integrity, and building the Docker container on every push to the repository. Docker containerization provides an isolated and reproducible environment.

**Q3: How are you handling numeric preprocessing for the predictions?**
**A3:** We use Scikit-Learn's `StandardScaler` to bring feature distributions to mean 0 and standard deviation 1. Both the model and the scaler are serialized using `joblib` and exported to the `models/` directory, ensuring user inputs from the UI are scaled identically to the training data.

**Q4: How did you handle categorical variables in the Heart Disease dataset?**
**A4:** We systematically mapped standard string categories (like 'ChestPainType': ATA, NAP, etc.) to numeric integers before passing them into the `StandardScaler` and models. This ensures compatibility with algorithms like Random Forest and SVM.

**Q5: Why did you choose Streamlit for the frontend?**
**A5:** Streamlit enables rapid prototyping of data applications directly in Python. It's perfectly suited for ML projects as it seamlessly integrates with Pandas and joblib, allowing us to build an interactive dashboard without needing a separate frontend framework.
