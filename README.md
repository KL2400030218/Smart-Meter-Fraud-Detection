# ⚡ AI-Enabled Smart Meter Fraud Detection and Blockchain Verification System

An integrated machine learning and blockchain-based system designed to detect potentially fraudulent smart meter readings, assess fraud risk, and provide tamper-evident verification of recorded fraud predictions.

## 📌 Overview

Smart meter fraud can lead to energy losses, inaccurate billing, and operational challenges for electricity distribution companies. This project combines machine learning techniques with blockchain technology to identify suspicious electricity consumption patterns and maintain the integrity of fraud detection records.

The system analyzes smart meter readings, generates fraud predictions and risk scores, stores prediction information in PostgreSQL, and allows selected fraud records to be recorded on an Ethereum-compatible blockchain using a Solidity smart contract.

The blockchain stores integrity-related metadata and a SHA-256 hash rather than raw meter readings. This enables verification of whether a stored fraud record has been modified after blockchain recording.

## 🎯 Objectives

- Detect potentially fraudulent smart meter readings using machine learning.
- Perform data preprocessing and feature engineering.
- Compare multiple ML models using evaluation metrics.
- Generate fraud probabilities, risk scores, and risk classifications.
- Provide an API for prediction and blockchain operations.
- Store meter readings and fraud predictions in PostgreSQL.
- Record fraud metadata on a blockchain using a smart contract.
- Verify record integrity through cryptographic hash comparison.
- Evaluate model performance, scalability, and blockchain transaction performance.

## 🏗️ System Architecture

```text
Smart Meter Readings
        |
        v
Data Cleaning and Preprocessing
        |
        v
Feature Engineering
        |
        v
ML Fraud Prediction
        |
        v
Fraud Probability, Risk Score, Risk Level
        |
        v
PostgreSQL Database
        |
        v
Fraud Record Selection
        |
        v
SHA-256 Hash Generation
        |
        v
FraudDetectionRegistry Smart Contract
        |
        v
Local Hardhat Blockchain
        |
        v
Hash Verification
        |
        v
React Dashboard
```

## 🧠 Machine Learning

The machine learning module includes preprocessing, model training, prediction, and evaluation.

### Models Compared

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- XGBoost
- LightGBM
- Neural Network (MLP)
- Proposed Soft-Voting Ensemble (Random Forest + XGBoost + LightGBM)

### Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Inference Latency
- Model Storage Size

The model comparison uses stratified 5-fold cross-validation.

The proposed ensemble combines the predicted probabilities of Random Forest, XGBoost, and LightGBM through soft voting.

### ML Module Features

- Data generation
- Data preprocessing
- Feature engineering
- Model training
- Model comparison
- Fraud prediction
- Risk scoring and classification
- Scalability testing

## ⛓️ Blockchain Integration

The project uses Solidity and Web3.py to integrate blockchain verification into the fraud detection workflow.

### Smart Contract

`FraudDetectionRegistry.sol`

### Smart Contract Functions

- `recordFraud()` – Records fraud-related metadata on-chain.
- `getFraudRecord()` – Retrieves a stored fraud record.
- `getRecordCount()` – Returns the number of recorded entries.
- `owner()` – Retrieves the contract owner.

### Blockchain Features

- SHA-256 hashing of canonical fraud records
- On-chain storage of fraud metadata
- Transaction receipt handling
- Blockchain record retrieval
- Database-to-blockchain hash verification
- Tamper detection

The blockchain component is deployed on a local Hardhat Ethereum-compatible development network for academic demonstration.

## 🗄️ Database

PostgreSQL is used for structured data storage.

### Database Tables

- `meters`
- `meter_readings`
- `fraud_predictions`
- `blockchain_records`

The database stores meter information, readings, prediction results, and blockchain transaction metadata.

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| Frontend | React, Vite |
| Backend | Python, FastAPI |
| Machine Learning | Scikit-learn, XGBoost, LightGBM |
| Database | PostgreSQL |
| Blockchain | Ethereum-compatible local network |
| Smart Contracts | Solidity |
| Blockchain Development | Hardhat |
| Blockchain Integration | Web3.py |
| Data Processing | Pandas, NumPy |
| API Documentation | FastAPI Swagger UI |
| Cryptographic Hashing | SHA-256 |

## 📁 Project Structure

```text
Smart-Meter-Fraud-Detection/
│
├── backend/
│   ├── app/
│   ├── requirements.txt
│   └── .env.example
│
├── frontend/
│   ├── src/
│   ├── package.json
│   └── package-lock.json
│
├── ml/
│   ├── data/
│   ├── models/
│   ├── notebooks/
│   ├── generate_dataset.py
│   ├── predict.py
│   ├── preprocess.py
│   ├── promote_production_model.py
│   ├── train.py
│   ├── train_v2.py
│   └── scalability_test.py
│
├── blockchain/
│   ├── contracts/
│   ├── scripts/
│   ├── test/
│   └── hardhat.config.js
│
├── database/
│   └── schema.sql
│
├── docs/
│
├── DEMO_SCRIPT.md
├── QUICKSTART.md
├── PROGRESS.md
├── README.md
└── .gitignore
```

## 🚀 Running the Project Locally

### Prerequisites

Install the following:

- Python 3.10 or compatible version
- Node.js and npm
- PostgreSQL
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/KL2400030218/Smart-Meter-Fraud-Detection.git
cd Smart-Meter-Fraud-Detection
```

### 2. Configure the Database

Create a PostgreSQL database and execute the SQL schema from:

```text
database/schema.sql
```

Configure the database connection using the backend `.env.example` as a template.

Create your own `.env` file with the appropriate local configuration values. Do not commit your actual `.env` file or credentials.

### 3. Start the Local Blockchain

Open a terminal at the project root and navigate to the blockchain directory:

```bash
cd blockchain
npm install
npx hardhat node
```

Keep the Hardhat node running.

### 4. Deploy the Smart Contract

Open a second terminal and navigate to the blockchain directory:

```bash
cd Smart-Meter-Fraud-Detection/blockchain
npx hardhat run scripts/deploy.js --network localhost
```

The deployment script deploys the `FraudDetectionRegistry` contract to the local Hardhat network.

Configure the backend with the required local blockchain connection and deployed contract details.

### 5. Start the Backend

Open another terminal at the project root:

```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Backend API:

```text
http://127.0.0.1:8000
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

### 6. Start the Frontend

Open another terminal at the project root:

```bash
cd frontend
npm install
npm run dev
```

Open the local frontend URL provided by Vite, typically:

```text
http://localhost:5173
```

Refer to `QUICKSTART.md` for the complete startup procedure and configuration details.

## 📊 Evaluation

The project includes model evaluation, scalability testing, and blockchain performance testing.

### Machine Learning Evaluation

The candidate ML models are evaluated using stratified 5-fold cross-validation with metrics such as F1-score, recall, precision, accuracy, ROC-AUC, latency, and model storage size.

### Scalability Evaluation

The scalability experiment evaluates processing performance across different meter counts and reading volumes, including:

- 100 meters
- 1,000 meters
- 10,000 meters
- 100,000 meters (attempted; execution timed out)
- 1 million readings

The scalability results are available in:

```text
ml/models/scalability_results.json
```

### Blockchain Evaluation

Blockchain transaction and verification performance is evaluated using a local Hardhat network.

The benchmark results are available in:

```text
ml/models/blockchain_throughput_results.json
```

## 🔐 Security and Integrity

The system uses SHA-256 hashing to generate an integrity hash for a canonical fraud record.

During verification:

1. The fraud record is retrieved from PostgreSQL.
2. Its canonical hash is recomputed.
3. The corresponding on-chain hash is retrieved.
4. The hashes are compared.
5. The verification result indicates whether the record matches or has been modified.

Raw smart meter readings are not stored directly on the blockchain.

## 🖥️ Application Features

The React dashboard provides interfaces for:

- Smart Meter Dashboard
- Meter Management
- Fraud Detection
- Fraud Records
- Blockchain Records
- Record Verification
- ML Model Information

## ⚠️ Limitations

- The current implementation uses a local Hardhat blockchain for academic demonstration.
- The application is not currently a publicly hosted production service.
- Scalability experiments use synthetic data for throughput testing.
- The 100,000-meter scalability test did not complete within the available execution time.
- The system is a research prototype and requires further validation using real-world utility datasets and a production blockchain environment.
- The adaptive blockchain-recording recommendation is calculated by the ML module, while blockchain recording is triggered separately through the backend API.

## 🔮 Future Enhancements

- Deployment of the frontend and backend to cloud platforms.
- Deployment of the smart contract to a public testnet.
- Integration with real-world smart meter datasets.
- Automated fraud alerts and notifications.
- Model monitoring and retraining.
- Improved blockchain access control.
- Advanced explainability techniques.
- Enhanced scalability and performance optimization.

## 👩‍💻 Project Information

**Project Title:** AI-Enabled Smart Meter Fraud Detection and Blockchain Verification System

**Domain:** Machine Learning, Blockchain, Web3, Smart Energy Systems

**Project Type:** Academic Research and Development Prototype

## 📄 Academic Use

This project is developed for academic and educational purposes.
