### FINOVA — Fintech Transaction Prediction
## FINOVA is a Python-based fintech machine learning application that provides a FastAPI backend and a Streamlit dashboard for exploring transaction data and generating transaction-amount predictions.
## Project status: The application has been containerized with Docker. The trained model is included in the repository. The source dataset and Jupyter notebook are intentionally excluded from GitHub.

## Features
- Machine learning inference: Loads a saved XGBoost model to produce transaction-amount predictions.
- FastAPI backend: Exposes the prediction application through an API and interactive API documentation.
- Streamlit dashboard: Provides a browser-based interface for interacting with the project.
- Docker support: Packages the application and its runtime dependencies in a container.
- Reproducible setup: Uses requirements.txt to define Python dependencies.

## Technology stack
- Python 3.13
- FastAPI and Uvicorn
- Streamlit
- XGBoost and scikit-learn
- Pandas / NumPy (where used by the application)
- Docker
- 
## Repository structure
fintech_transaction_project/
├── main/
│   ├── app.py
│   └── streamlit_app.py
├── model/
│   └── xgboost_transaction_amount_model.joblib
├── .dockerignore
├── .gitignore
├── dockerfile
└── requirements.txt
The local data/ directory and notebook/ directory are not tracked in Git. They remain local because the source CSV and notebook are large files and are not required to browse the source code on GitHub. The Docker image was built separately with the files included in its build context.
## Run locally
1. Clone the repository
git clone https://github.com/faithiskey/Fintech_transaction.git
cd Fintech_transaction
2. Create and activate a virtual environment
Windows PowerShell:
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
If PowerShell blocks activation for this session, run:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
3. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
4. Start the FastAPI API
From the project root:
uvicorn main.app:app --host 0.0.0.0 --port 8000
Open:
- API documentation: http://localhost:8000/docs
- Alternative API documentation: http://localhost:8000/redoc
5. Start the Streamlit dashboard
Open a second terminal in the project root, activate the same virtual environment, and run:
streamlit run main/streamlit_app.py --server.port 8501
Open the dashboard at http://localhost:8501.
Note: Run the commands from the repository root so the main and model paths resolve as expected. The model file must be present at the path expected by the application. If the app relies on the excluded CSV at runtime, obtain that dataset separately and place it in the expected local data/ path.

Run with Docker
The project includes a dockerfile that installs the dependencies, copies the application and model, and starts the API and dashboard in a container.
Build the image
Run this from the project root:
docker build -t iwuchukwu/fintech_transaction:version1 -f dockerfile .
The Docker build context must include any files the Dockerfile copies, including data/ if the Dockerfile still contains COPY data/ ./data/. The CSV and notebook are excluded from Git, but the local dataset can still be used for a local Docker build if it is present and not excluded by .dockerignore.
Start the container
docker run -d --name finova -p 8000:8000 -p 8501:8501 iwuchukwu/fintech_transaction:version1
Then open:
- Streamlit dashboard: http://localhost:8501
- FastAPI docs: http://localhost:8000/docs
Useful Docker commands
# View running containers
docker ps

# View application logs
docker logs finova

# Follow logs as the app runs
docker logs -f finova

# Stop the container
docker stop finova

# Start it again
docker start finova

# Remove the container (stop it first if it is running)
docker rm -f finova
Pull the published image
If the image has been published to Docker Hub under this repository and tag, another machine can retrieve it with:
docker pull faithonaluiwuchukwu/fintech_transaction:version1
Then run it:
docker run -d --name finova -p 8000:8000 -p 8501:8501 faithonaluiwuchukwu/fintech_transaction:version1
Docker Hub repository: https://hub.docker.com/r/faithonaluiwuchukwu/fintech_transaction
Model evaluation and limitations
The current experiment produced the following reported evaluation results:
Model	MAE	RMSE	R²
Linear Regression	13,075,509.50	15,159,973.95	0.0049
XGBoost	13,128,583.22	15,199,648.18	-0.0003


These results indicate very weak predictive performance on the evaluated data. The XGBoost model's negative R² means it performed worse than a baseline that predicts the target mean on that evaluation set. Treat predictions as experimental, not as reliable financial estimates or a production decision signal.

Recommended next steps include checking target leakage and feature quality, evaluating an appropriate baseline, reviewing train/test splitting, considering transaction type and currency differences, and validating performance on held-out or future data. Error magnitudes should also be interpreted in the target currency and units used by the dataset.
Data and notebook availability
The following are intentionally excluded from this GitHub repository:
- data/ — source transaction CSV files
- notebook/ and *.ipynb — exploratory and model-development notebooks
This keeps large source files out of Git history. The trained model artifact is included. Users who want to reproduce training or data analysis will need to obtain the dataset separately and follow any applicable data licensing and privacy requirements.
Security and deployment notes
- Do not commit API keys, credentials, .env files, or sensitive transaction data.
- The Docker command above publishes ports on the host. Before exposing the app to the public internet, configure appropriate network rules, authentication where needed, HTTPS, and production process management.
- The application currently starts Uvicorn and Streamlit in the same container. For a more robust production deployment, consider running them as separate services or containers with health checks and managed logging.
- Verify that the scikit-learn version used to load the saved model is compatible with the version used when the model was serialized. Version mismatches can produce warnings or unexpected behavior.
Future improvements
- Improve model evaluation and prediction quality.
- Add automated tests for API and prediction behavior.
- Add container health checks and structured logging.
- Deploy the services to a cloud platform.
- Add CI checks for linting, tests, and Docker builds.

Author
Faith Onalu-Iwuchukwu
- GitHub: https://github.com/faithiskey
- Portfolio: https://datascienceportfol.io/faithchiiwu
- LinkedIn: https://www.linkedin.com/in/faith-onalu-iwuchukwu-798a591ab/
