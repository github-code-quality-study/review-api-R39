import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from urllib.parse import parse_qs, urlparse
import json
import pandas as pd
from datetime import datetime
import uuid
import os
from typing import Callable, Any
from wsgiref.simple_server import make_server

nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)

sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

reviews = pd.read_csv('data/reviews.csv').to_dict('records')

class ReviewAnalyzerServer:
    def __init__(self) -> None:
        self.reviews = reviews

    def analyze_sentiment(self, review_body):
        sentiment_scores = sia.polarity_scores(review_body)
        return sentiment_scores

    def __call__(self, environ: dict[str, Any], start_response: Callable[..., Any]) -> bytes:
        if environ["REQUEST_METHOD"] == "GET":
            query_params = parse_qs(environ["QUERY_STRING"])
            location = query_params.get("location", [None])[0]
            start_date = query_params.get("start_date", [None])[0]
            end_date = query_params.get("end_date", [None])[0]

            filtered_reviews = self.filter_reviews(location, start_date, end_date)
            for review in filtered_reviews:
                review['sentiment'] = self.analyze_sentiment(review['ReviewBody'])

            sorted_reviews = sorted(filtered_reviews, key=lambda x: x['sentiment']['compound'], reverse=True)
            response_body = json.dumps(sorted_reviews, indent=2).encode("utf-8")
            
            start_response("200 OK", [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(response_body)))
            ])
            
            return [response_body]

        if environ["REQUEST_METHOD"] == "POST":
            try:
                size = int(environ.get('CONTENT_LENGTH', 0))
                data = environ['wsgi.input'].read(size).decode('utf-8')
                post_params = parse_qs(data)
                location = post_params.get("Location", [None])[0]
                review_body = post_params.get("ReviewBody", [None])[0]

                if not location or not review_body:
                    start_response("400 Bad Request", [("Content-Type", "application/json")])
                    return [b'{"error": "Location and ReviewBody are required"}']

                if location not in [
                    "Albuquerque, New Mexico", "Carlsbad, California", "Chula Vista, California", "Colorado Springs, Colorado",
                    "Denver, Colorado", "El Cajon, California", "El Paso, Texas", "Escondido, California", "Fresno, California",
                    "La Mesa, California", "Las Vegas, Nevada", "Los Angeles, California", "Oceanside, California", "Phoenix, Arizona",
                    "Sacramento, California", "Salt Lake City, Utah", "San Diego, California", "Tucson, Arizona"
                ]:
                    start_response("400 Bad Request", [("Content-Type", "application/json")])
                    return [b'{"error": "Invalid location"}']

                new_review = {
                    "ReviewId": str(uuid.uuid4()),
                    "Location": location,
                    "Timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "ReviewBody": review_body
                }

                self.reviews.append(new_review)
                response_body = json.dumps(new_review, indent=2).encode("utf-8")
                start_response("201 Created", [
                    ("Content-Type", "application/json"),
                    ("Content-Length", str(len(response_body)))
                ])
                
                return [response_body]

            except Exception as e:
                start_response("500 Internal Server Error", [("Content-Type", "application/json")])
                return [b'{"error": "Internal Server Error"}']

    def filter_reviews(self, location=None, start_date=None, end_date=None):
        filtered_reviews = self.reviews

        if location:
            filtered_reviews = [r for r in filtered_reviews if r["Location"] == location]
        
        if start_date:
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
            filtered_reviews = [r for r in filtered_reviews if datetime.strptime(r["Timestamp"], '%Y-%m-%d %H:%M:%S') >= start_date]

        if end_date:
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
            filtered_reviews = [r for r in filtered_reviews if datetime.strptime(r["Timestamp"], '%Y-%m-%d %H:%M:%S') <= end_date]

        return filtered_reviews

if __name__ == "__main__":
    app = ReviewAnalyzerServer()
    port = int(os.environ.get('PORT', 8080))
    with make_server("", port, app) as httpd:
        print(f"Listening on port {port}...")
        httpd.serve_forever()
