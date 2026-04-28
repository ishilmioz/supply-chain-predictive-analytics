import pandas as pd
import joblib
import streamlit as st
import os

@st.cache_resource
def load_prediction_model(model_path):
    """Modeli güvenli bir şekilde yükler."""
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

def prepare_input_data(product_price, item_quantity, shipping_days_map, shipping_mode, 
                       order_hour, order_date, payment_type, customer_segment, 
                       order_region, order_country_es, category_id, discount_rate): # discount_rate eklendi
    
    scheduled_days = shipping_days_map[shipping_mode]
    time_factor = 4 / (scheduled_days + 1)
    sales = product_price * item_quantity
    
    load_risk_score = sales * time_factor
    temporal_stress = int(order_hour > 17) * time_factor
    order_month = order_date.month
    order_day_of_week = order_date.weekday()
    is_weekend = 1 if order_day_of_week >= 5 else 0

    data = {
        'product_price': product_price, 'item_quantity': item_quantity, 'sales': sales,
        'discount_rate': discount_rate, 
        'load_risk_score': load_risk_score, 'temporal_stress': temporal_stress,
        'payment_type': payment_type, 'customer_segment': customer_segment,
        'order_region': order_region, 'order_country': order_country_es,
        'order_day_of_week': order_day_of_week, 'order_month': order_month,
        'order_hour': order_hour, 'is_weekend': is_weekend, 'category_id': category_id
    }
    return pd.DataFrame([data])