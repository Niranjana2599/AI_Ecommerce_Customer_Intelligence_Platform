import os
import time
import nltk

from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel

from prometheus_client import (
    Counter,
    Histogram,
    generate_latest
)

# =========================================================
# NLTK — download only if missing
# =========================================================

def _ensure_nltk_data():

    resources = {
        "punkt": "tokenizers/punkt",
        "punkt_tab": "tokenizers/punkt_tab",
        "stopwords": "corpora/stopwords",
        "wordnet": "corpora/wordnet",
        "averaged_perceptron_tagger": "taggers/averaged_perceptron_tagger",
        "omw-1.4": "corpora/omw-1.4",
    }

    for name, path in resources.items():

        try:
            nltk.data.find(path)

        except LookupError:
            nltk.download(name, quiet=True)

_ensure_nltk_data()

# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(
    title="AI Ecommerce Customer Intelligence API",
    description="Churn · CLV · Delay · Recommendations · NLP · LightGCN · RAG",
    version="1.0.0",
)

# =========================================================
# PROMETHEUS METRICS
# =========================================================

REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total API Requests",
    ["method", "endpoint"]
)

REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "API Request Latency",
    ["endpoint"]
)

# =========================================================
# MONITORING MIDDLEWARE
# =========================================================

@app.middleware("http")
async def monitor_requests(request, call_next):

    start_time = time.time()

    response = await call_next(request)

    duration = time.time() - start_time

    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path
    ).inc()

    REQUEST_LATENCY.labels(
        endpoint=request.url.path
    ).observe(duration)

    return response

# =========================================================
# LAZY MODEL REGISTRY
# =========================================================

_predict_churn_fn = None
_predict_clv_fn = None
_predict_delay_fn = None
_recommend_products_fn = None
_predict_sentiment_fn = None
_summarize_review_fn = None
_detect_complaint_fn = None
_lightgcn_recommend_fn = None
_ask_rag_fn = None


def _get_predict_churn():
    global _predict_churn_fn

    if _predict_churn_fn is None:
        from src.ml.churn.predict import predict_churn
        _predict_churn_fn = predict_churn

    return _predict_churn_fn


def _get_predict_clv():
    global _predict_clv_fn

    if _predict_clv_fn is None:
        from src.ml.clv.predict import predict_clv
        _predict_clv_fn = predict_clv

    return _predict_clv_fn


def _get_predict_delay():
    global _predict_delay_fn

    if _predict_delay_fn is None:
        from src.ml.delay_prediction.predict import predict_delay
        _predict_delay_fn = predict_delay

    return _predict_delay_fn


def _get_recommend_products():
    global _recommend_products_fn

    if _recommend_products_fn is None:
        from src.ml.recommendations.predict import recommend_products
        _recommend_products_fn = recommend_products

    return _recommend_products_fn


def _get_predict_sentiment():
    global _predict_sentiment_fn

    if _predict_sentiment_fn is None:
        from src.nlp.sentiment_analysis import predict_sentiment
        _predict_sentiment_fn = predict_sentiment

    return _predict_sentiment_fn


def _get_summarize_review():
    global _summarize_review_fn

    if _summarize_review_fn is None:
        from src.nlp.summarization import summarize_review
        _summarize_review_fn = summarize_review

    return _summarize_review_fn


def _get_detect_complaint():
    global _detect_complaint_fn

    if _detect_complaint_fn is None:
        from src.nlp.complaint_detection import detect_complaint
        _detect_complaint_fn = detect_complaint

    return _detect_complaint_fn


def _get_lightgcn_recommend():
    global _lightgcn_recommend_fn

    if _lightgcn_recommend_fn is None:
        from src.dl.lightgcn.predict import recommend_products as lightgcn_recommend
        _lightgcn_recommend_fn = lightgcn_recommend

    return _lightgcn_recommend_fn


def _get_ask_rag():
    global _ask_rag_fn

    if _ask_rag_fn is None:
        from src.rag.predict import ask_rag
        _ask_rag_fn = ask_rag

    return _ask_rag_fn

# =========================================================
# INPUT SCHEMAS
# =========================================================

class EcommerceInput(BaseModel):

    order_status: str

    delivery_duration_days: float
    approval_time_hours: float

    purchase_month: int
    purchase_year: int
    purchase_weekday: int

    customer_gender: str
    customer_age: int
    customer_zip_code_prefix: int
    customer_city: str
    customer_state: str
    customer_segment: int

    order_item_id: int

    price_x: float
    freight_value: float
    discount_rate: float

    product_category_name: int
    product_name: str
    product_brand: int

    product_weight_g: float
    product_length_cm: float
    product_height_cm: float
    product_width_cm: float

    cost: float
    price_y: float

    seller_contact_gender: str
    seller_contact_age: int
    seller_zip_code_prefix: int
    seller_city: str
    seller_state: str

    payment_sequential: int
    payment_type: int
    payment_installments: int
    payment_value: float

    review_score: int

    purchase_day: int
    purchase_hour: int
    purchase_weekend: int

    payment_apple_pay: int
    payment_bank_transfer: int
    payment_boleto: int
    payment_credit_card: int
    payment_debit_card: int
    payment_paypal: int
    payment_voucher: int

    payment_apple_pay_1: int
    payment_bank_transfer_1: int
    payment_boleto_1: int
    payment_credit_card_1: int
    payment_debit_card_1: int
    payment_paypal_1: int
    payment_voucher_1: int

    review_length: int
    review_word_count: int

    total_orders: int
    total_spent: float

    avg_order_value: float
    avg_review_score: float

    avg_delivery_delay: float

    unique_products_purchased: int

    Recency: int
    Frequency: int
    Monetary: float


class RecommendationInput(BaseModel):
    product_id: str


class NLPInput(BaseModel):
    text: str


class LightGCNInput(BaseModel):
    customer_id: str
    top_k: int = 10


class RagInput(BaseModel):
    question: str

# =========================================================
# HEALTH ROUTE
# =========================================================

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }

# =========================================================
# METRICS ROUTE
# =========================================================

@app.get("/metrics")
def metrics():

    return Response(
        generate_latest(),
        media_type="text/plain"
    )

# =========================================================
# HOME ROUTE
# =========================================================

@app.get("/")
def home():

    return {
        "message": "AI Ecommerce Customer Intelligence API Running Successfully"
    }

# =========================================================
# CHURN PREDICTION
# =========================================================

@app.post("/predict/churn")
def churn_prediction(data: EcommerceInput):

    result = _get_predict_churn()(data.dict())

    return result

# =========================================================
# CLV PREDICTION
# =========================================================

@app.post("/predict/clv")
def clv_prediction(data: EcommerceInput):

    result = _get_predict_clv()(data.dict())

    return result

# =========================================================
# DELIVERY DELAY PREDICTION
# =========================================================

@app.post("/predict/delay")
def delay_prediction(data: EcommerceInput):

    result = _get_predict_delay()(data.dict())

    return result

# =========================================================
# PRODUCT RECOMMENDATION
# =========================================================

@app.post("/recommend/products")
def recommendation_endpoint(data: RecommendationInput):

    result = _get_recommend_products()(data.product_id)

    return result

# =========================================================
# SENTIMENT ANALYSIS
# =========================================================

@app.post("/nlp/sentiment")
def sentiment_route(data: NLPInput):

    result = _get_predict_sentiment()(data.text)

    return result

# =========================================================
# REVIEW SUMMARIZATION
# =========================================================

@app.post("/nlp/summarize")
def summarize_route(data: NLPInput):

    result = _get_summarize_review()(data.text)

    return {
        "summary": result
    }

# =========================================================
# COMPLAINT DETECTION
# =========================================================

@app.post("/nlp/complaint")
def complaint_route(data: NLPInput):

    result = _get_detect_complaint()(data.text)

    return result

# =========================================================
# LIGHTGCN RECOMMENDATION
# =========================================================

@app.post("/recommend/lightgcn")
def lightgcn_route(data: LightGCNInput):

    result = _get_lightgcn_recommend()(
        data.customer_id,
        data.top_k
    )

    return result

# =========================================================
# RAG CHATBOT
# =========================================================

@app.post("/rag/chat")
def rag_chatbot(data: RagInput):

    result = _get_ask_rag()(data.question)

    return result