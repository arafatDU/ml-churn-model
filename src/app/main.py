"""
FASTAPI + GRADIO SERVING APPLICATION - Production-Ready ML Model Serving
========================================================================

This application provides a complete serving solution for the Telco Customer Churn model
with both programmatic API access and a user-friendly web interface.

Architecture:
- FastAPI: High-performance REST API with automatic OpenAPI documentation
- Gradio: User-friendly web UI for manual testing and demonstrations
- Pydantic: Data validation and automatic API documentation
"""

from fastapi import FastAPI
from pydantic import BaseModel
import gradio as gr
from src.serving.inference import predict  # Core ML inference logic

# Initialize FastAPI application
app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description="ML API for predicting customer churn in telecom industry",
    version="1.0.0"
)

# === HEALTH CHECK ENDPOINT ===
# CRITICAL: Required for AWS Application Load Balancer health checks
@app.get("/")
def root():
    """
    Health check endpoint for monitoring and load balancer health checks.
    """
    return {"status": "ok"}

# === REQUEST DATA SCHEMA ===
# Pydantic model for automatic validation and API documentation
class CustomerData(BaseModel):
    """
    Customer data schema for churn prediction.
    
    This schema defines the exact 18 features required for churn prediction.
    All features match the original dataset structure for consistency.
    """
    # Demographics
    gender: str                # "Male" or "Female"
    Partner: str               # "Yes" or "No" - has partner
    Dependents: str            # "Yes" or "No" - has dependents
    
    # Phone services
    PhoneService: str          # "Yes" or "No"
    MultipleLines: str         # "Yes", "No", or "No phone service"
    
    # Internet services  
    InternetService: str       # "DSL", "Fiber optic", or "No"
    OnlineSecurity: str        # "Yes", "No", or "No internet service"
    OnlineBackup: str          # "Yes", "No", or "No internet service"
    DeviceProtection: str      # "Yes", "No", or "No internet service"
    TechSupport: str           # "Yes", "No", or "No internet service"
    StreamingTV: str           # "Yes", "No", or "No internet service"
    StreamingMovies: str       # "Yes", "No", or "No internet service"
    
    # Account information
    Contract: str              # "Month-to-month", "One year", "Two year"
    PaperlessBilling: str      # "Yes" or "No"
    PaymentMethod: str         # "Electronic check", "Mailed check", etc.
    
    # Numeric features
    tenure: int                # Number of months with company
    MonthlyCharges: float      # Monthly charges in dollars
    TotalCharges: float        # Total charges to date

# === MAIN PREDICTION API ENDPOINT ===
@app.post("/predict")
def get_prediction(data: CustomerData):
    """
    Main prediction endpoint for customer churn prediction.
    
    This endpoint:
    1. Receives validated customer data via Pydantic model
    2. Calls the inference pipeline to transform features and predict
    3. Returns churn prediction in JSON format
    
    Expected Response:
    - {"prediction": "Likely to churn"} or {"prediction": "Not likely to churn"}
    - {"error": "error_message"} if prediction fails
    """
    try:
        # Convert Pydantic model to dict and call inference pipeline
        result = predict(data.dict())
        return {"prediction": result}
    except Exception as e:
        # Return error details for debugging (consider logging in production)
        return {"error": str(e)}


# =================================================== # 


# === GRADIO WEB INTERFACE ===
def gradio_interface(
    gender, Partner, Dependents, PhoneService, MultipleLines,
    InternetService, OnlineSecurity, OnlineBackup, DeviceProtection,
    TechSupport, StreamingTV, StreamingMovies, Contract,
    PaperlessBilling, PaymentMethod, tenure, MonthlyCharges, TotalCharges
):
    """
    Gradio interface function that processes form inputs and returns prediction.
    
    This function:
    1. Takes individual form inputs from Gradio UI
    2. Constructs the data dictionary matching the API schema
    3. Calls the same inference pipeline used by the API
    4. Returns user-friendly prediction string
    
    """
    # Construct data dictionary matching CustomerData schema
    data = {
        "gender": gender,
        "Partner": Partner,
        "Dependents": Dependents,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "tenure": int(tenure),              # Ensure integer type
        "MonthlyCharges": float(MonthlyCharges),  # Ensure float type
        "TotalCharges": float(TotalCharges),      # Ensure float type
    }
    
    # Call same inference pipeline as API endpoint
    result = predict(data)
    return str(result)  # Return as string for Gradio display

# === GRADIO UI CONFIGURATION ===
# Build comprehensive Gradio interface with improved layout using Blocks
with gr.Blocks(
    title="Telco Churn Predictor",
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate"
    ),
    css="""
        .gradio-container {max-width: 1200px !important}
        .output-text {font-size: 18px; font-weight: bold;}
        .prediction-high {color: #dc2626; background-color: #fef2f2; padding: 15px; border-radius: 8px;}
        .prediction-low {color: #16a34a; background-color: #f0fdf4; padding: 15px; border-radius: 8px;}
    """
) as demo:
    
    gr.Markdown("""
    # 🔮 Telco Customer Churn Predictor
    
    Predict customer churn probability using advanced machine learning (XGBoost). 
    Fill in the customer details below to identify customers at risk of leaving.
    
    💡 **Key Risk Factors**: Month-to-month contracts, fiber optic internet, electronic check payments, and short tenure.
    """)
    
    with gr.Tabs():
        with gr.Tab("📊 Predict Churn"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 👤 Demographics")
                    gender = gr.Dropdown(["Male", "Female"], label="Gender", value="Female")
                    Partner = gr.Dropdown(["Yes", "No"], label="Has Partner", value="No")
                    Dependents = gr.Dropdown(["Yes", "No"], label="Has Dependents", value="No")
                    
                    gr.Markdown("### 📞 Phone Services")
                    PhoneService = gr.Dropdown(["Yes", "No"], label="Phone Service", value="Yes")
                    MultipleLines = gr.Dropdown(["Yes", "No", "No phone service"], label="Multiple Lines", value="No")
                
                with gr.Column(scale=1):
                    gr.Markdown("### 🌐 Internet & Add-ons")
                    InternetService = gr.Dropdown(["DSL", "Fiber optic", "No"], label="Internet Service", value="Fiber optic")
                    OnlineSecurity = gr.Dropdown(["Yes", "No", "No internet service"], label="Online Security", value="No")
                    OnlineBackup = gr.Dropdown(["Yes", "No", "No internet service"], label="Online Backup", value="No")
                    DeviceProtection = gr.Dropdown(["Yes", "No", "No internet service"], label="Device Protection", value="No")
                    TechSupport = gr.Dropdown(["Yes", "No", "No internet service"], label="Tech Support", value="No")
                    StreamingTV = gr.Dropdown(["Yes", "No", "No internet service"], label="Streaming TV", value="Yes")
                    StreamingMovies = gr.Dropdown(["Yes", "No", "No internet service"], label="Streaming Movies", value="Yes")
                
                with gr.Column(scale=1):
                    gr.Markdown("### 💳 Billing & Contract")
                    Contract = gr.Dropdown(["Month-to-month", "One year", "Two year"], label="Contract Type", value="Month-to-month")
                    PaperlessBilling = gr.Dropdown(["Yes", "No"], label="Paperless Billing", value="Yes")
                    PaymentMethod = gr.Dropdown([
                        "Electronic check", 
                        "Mailed check",
                        "Bank transfer (automatic)", 
                        "Credit card (automatic)"
                    ], label="Payment Method", value="Electronic check")
                    
                    gr.Markdown("### 💰 Account Information")
                    tenure = gr.Slider(minimum=0, maximum=100, value=1, step=1, label="Tenure (months)")
                    MonthlyCharges = gr.Slider(minimum=0, maximum=200, value=85.0, step=0.1, label="Monthly Charges ($)")
                    TotalCharges = gr.Number(label="Total Charges ($)", value=85.0, minimum=0)
            
            with gr.Row():
                predict_btn = gr.Button("🎯 Predict Churn Risk", variant="primary", size="lg")
                clear_btn = gr.ClearButton(components=[
                    gender, Partner, Dependents, PhoneService, MultipleLines,
                    InternetService, OnlineSecurity, OnlineBackup, DeviceProtection,
                    TechSupport, StreamingTV, StreamingMovies, Contract,
                    PaperlessBilling, PaymentMethod, tenure, MonthlyCharges, TotalCharges
                ], value="🔄 Reset")
            
            output = gr.Markdown(label="Prediction Result", elem_classes="output-text")
            
            def predict_with_styling(*args):
                """Enhanced prediction with visual styling"""
                result = gradio_interface(*args)
                
                # Parse result and add styling
                if "Will CHURN" in result or "High Risk" in result:
                    return f'<div class="prediction-high">⚠️ {result}</div>'
                else:
                    return f'<div class="prediction-low">✅ {result}</div>'
            
            predict_btn.click(
                fn=predict_with_styling,
                inputs=[
                    gender, Partner, Dependents, PhoneService, MultipleLines,
                    InternetService, OnlineSecurity, OnlineBackup, DeviceProtection,
                    TechSupport, StreamingTV, StreamingMovies, Contract,
                    PaperlessBilling, PaymentMethod, tenure, MonthlyCharges, TotalCharges
                ],
                outputs=output
            )
        
        with gr.Tab("📋 Example Scenarios"):
            gr.Markdown("""
            ### High Risk Customer Profile
            - **Contract**: Month-to-month
            - **Internet**: Fiber optic with no add-ons (security, backup, etc.)
            - **Payment**: Electronic check
            - **Tenure**: New customer (< 6 months)
            - **Characteristics**: No partner, no dependents, high monthly charges
            
            ### Low Risk Customer Profile
            - **Contract**: Two year contract
            - **Internet**: DSL with security add-ons
            - **Payment**: Automatic (bank transfer or credit card)
            - **Tenure**: Long-term customer (> 24 months)
            - **Characteristics**: Has partner and dependents, moderate charges
            """)
            
            gr.Examples(
                examples=[
                    ["Female", "No", "No", "Yes", "No", "Fiber optic", "No", "No", "No", 
                     "No", "Yes", "Yes", "Month-to-month", "Yes", "Electronic check", 
                     1, 85.0, 85.0],
                    ["Male", "Yes", "Yes", "Yes", "Yes", "DSL", "Yes", "Yes", "Yes",
                     "Yes", "No", "No", "Two year", "No", "Credit card (automatic)",
                     60, 45.0, 2700.0],
                    ["Female", "Yes", "No", "Yes", "No", "Fiber optic", "Yes", "Yes", "No",
                     "Yes", "Yes", "No", "One year", "Yes", "Bank transfer (automatic)",
                     24, 70.0, 1680.0]
                ],
                inputs=[
                    gender, Partner, Dependents, PhoneService, MultipleLines,
                    InternetService, OnlineSecurity, OnlineBackup, DeviceProtection,
                    TechSupport, StreamingTV, StreamingMovies, Contract,
                    PaperlessBilling, PaymentMethod, tenure, MonthlyCharges, TotalCharges
                ],
                label="Try these example customers"
            )
        
        with gr.Tab("ℹ️ Model Info"):
            gr.Markdown("""
            ### Model Details
            - **Algorithm**: XGBoost (Gradient Boosting)
            - **Features**: 30 engineered features from customer data
            - **Performance**: 
              - Recall: 83.2% (catches 83% of churners)
              - ROC AUC: 0.838 (excellent discrimination)
            - **Training Data**: 7,000+ telecom customers
            
            ### How to Use
            1. Fill in all customer information fields
            2. Click "🎯 Predict Churn Risk"
            3. Review the prediction result
            4. Take appropriate retention actions for high-risk customers
            
            ### API Access
            Use `/predict` endpoint for programmatic access:
            ```bash
            curl -X POST "http://localhost:8000/predict" \\
              -H "Content-Type: application/json" \\
              -d '{"gender": "Female", "Partner": "No", ...}'
            ```
            
            📖 Full API documentation: [/docs](/docs)
            """)
    
    gr.Markdown("""
    ---
    <div style="text-align: center; color: #666;">
    Built with FastAPI & Gradio | Model trained with MLflow
    </div>
    """)


# === MOUNT GRADIO UI INTO FASTAPI ===
# This creates the /ui endpoint that serves the Gradio interface
# IMPORTANT: This must be the final line to properly integrate Gradio with FastAPI
app = gr.mount_gradio_app(
    app,           # FastAPI application instance
    demo,          # Gradio interface
    path="/ui"     # URL path where Gradio will be accessible
)