# Sentiment-Based Product Recommendation System

This project is a sentiment-based product recommendation system that suggests top products to users based on their past reviews and sentiment analysis. It leverages machine learning models to analyze user sentiments and recommend products accordingly.

## Features

- Recommends top 5 products for a given user based on sentiment analysis.
- Uses pre-trained models and vectorizers stored in `pickle_files/`.
- Web interface for easy interaction ([templates/index.html](templates/index.html)).
- Sample data and attribute descriptions provided in the `data/` directory.

## Project Structure

- **app.py**: Main Flask application.
- **model.py**: Contains core recommendation logic.
- **pickle_files/**: Pre-trained models and vectorizers.
- **templates/index.html**: Web UI.
- **data/**: Sample datasets.

## Setup Instructions

### Step 1: Clone the repository

```bash
git clone https://github.com/Pavani89/capstone_project-sentiment-based-product-recommendation-system.git
```

### Step 2: Install dependencies

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

### Step 3: Load models and data

Ensure the following pickle files are in the `pickle_file` directory:
- `model.pkl`: Sentiment analysis model.
- `user_final_rating.pkl`: Collaborative filtering rating model.
- `count_vector.pkl`: CountVectorizer object.
- `tfidf_transformer.pkl`: TF-IDF transformer.
- `RandomForest_classifier.pkl`: Checkpoint file for randome forest model.

Also, ensure the `sample30.csv` dataset under the folder `data` is available for testing.

### Step 4: Run the Flask application

Start the Flask development server:

```bash
python app.py
```

Open a browser and navigate to `http://127.0.0.1:5000/` to view the app.

---

## Usage

- Enter a username (e.g., `walker557`, `kimmie`, `rebecca`) in the web interface and click "Get Recommendations" to view the top 5 recommended products.

## Data

- `data/sample30.csv`: Sample user reviews.
- `data/Data+Attribute+Description.csv`: Description of dataset attributes.

---

## Model Details

### 1. Sentiment Analysis

- **Input**: Cleaned review text.
- **Model**: Logistic Regression using TF-IDF features.
- **Output**: Sentiment classification (Positive, Negative, Neutral).
- **Metrics**: Accuracy, Precision, Recall, F1-Score.

### 2. User-User Collaborative Filtering

- **Input**: User-product interaction matrix.
- **Technique**: Cosine similarity between users.
- **Output**: Estimated ratings for products not yet rated.
- **Metric**: RMSE.

### Integration

The final recommendation engine merges sentiment filtering with collaborative filtering predictions. It ranks products by their predicted rating and sentiment polarity.

---

## Models

Pre-trained models and vectorizers are stored in the `pickle_files/` directory:
- `count_vector.pkl`
- `tfidf_transformer.pkl`
- `LR_classifier.pkl`
- `RF_classifier.pkl`
- `XGB_classifier.pkl`
- `model.pkl`
- `item_final_rating.pkl`
- `user_final_rating.pkl`

## License

This project is for educational purposes.