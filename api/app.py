from fastapi import FastAPI
from pydantic import BaseModel

from src.ml.churn.predict import predict_churn
from src.ml.clv.predict import predict_clv
from src.ml.delay_prediction.predict import predict_delay

from src.ml.recommendations.predict import (
    recommend_products
)

from src.nlp.sentiment_analysis import (
    predict_sentiment
)

from src.nlp.summarization import (
    summarize_review
)

from src.nlp.complaint_detection import (
    detect_complaint
)

from src.dl.lightgcn.predict import (
    recommend_products as lightgcn_recommend
)

from src.rag.predict import ask_rag


# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(
    title="AI Ecommerce Customer Intelligence API"
)


# =========================================================
# COMMON INPUT SCHEMA
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


# =========================================================
# RECOMMENDATION INPUT
# =========================================================

class RecommendationInput(BaseModel):

    product_id: str


# =========================================================
# NLP INPUT
# =========================================================

class NLPInput(BaseModel):

    text: str


# =========================================================
# LIGHTGCN INPUT
# =========================================================

class LightGCNInput(BaseModel):

    customer_id: str

    top_k: int = 10


# =========================================================
# RAG INPUT
# =========================================================

class RagInput(BaseModel):

    question: str


# =========================================================
# HOME ROUTE
# =========================================================

@app.get("/")
def home():

    return {
        "message": "AI Ecommerce Customer Intelligence API Running Successfully"
    }


# =========================================================
# CHURN PREDICTION ROUTE
# =========================================================

@app.post("/predict/churn")
def churn_prediction(data: EcommerceInput):

    input_data = data.dict()

    result = predict_churn(input_data)

    return result


# =========================================================
# CLV PREDICTION ROUTE
# =========================================================

@app.post("/predict/clv")
def clv_prediction(data: EcommerceInput):

    input_data = data.dict()

    result = predict_clv(input_data)

    return result


# =========================================================
# DELIVERY DELAY PREDICTION ROUTE
# =========================================================

@app.post("/predict/delay")
def delay_prediction(data: EcommerceInput):

    input_data = data.dict()

    result = predict_delay(input_data)

    return result


# =========================================================
# PRODUCT RECOMMENDATION ROUTE
# =========================================================

@app.post("/recommend/products")
def recommendation_endpoint(data: RecommendationInput):

    result = recommend_products(data.product_id)

    return result


# =========================================================
# SENTIMENT ANALYSIS ROUTE
# =========================================================

@app.post("/nlp/sentiment")
def sentiment_route(data: NLPInput):

    result = predict_sentiment(data.text)

    return result


# =========================================================
# REVIEW SUMMARIZATION ROUTE
# =========================================================

@app.post("/nlp/summarize")
def summarize_route(data: NLPInput):

    result = summarize_review(data.text)

    return {

        "summary": result

    }


# =========================================================
# COMPLAINT DETECTION ROUTE
# =========================================================

@app.post("/nlp/complaint")
def complaint_route(data: NLPInput):

    result = detect_complaint(data.text)

    return result


# =========================================================
# LIGHTGCN RECOMMENDATION ROUTE
# =========================================================

@app.post("/recommend/lightgcn")
def lightgcn_route(data: LightGCNInput):

    result = lightgcn_recommend(

        data.customer_id,

        data.top_k

    )

    return result


# =========================================================
# RAG CHATBOT ENDPOINT
# =========================================================

@app.post("/rag/chat")
def rag_chatbot(data: RagInput):

    result = ask_rag(

        data.question

    )

    return result