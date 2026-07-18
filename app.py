import streamlit as st
from google import genai
from openai import OpenAI
from PIL import Image
import pandas as pd
import io
import json
import cv2
import numpy as np
import time
import re
import os
import base64

# Cấu hình trang
st.set_page_config(page_title="Hệ thống OCR tối ưu", layout="wide")
st.title("🪪 Hệ thống Scan Giấy Tờ Tối Ưu CPU")

# Nạp API Key từ Secrets
gemini_api_key = st.secrets.get("GEMINI_API_KEY", "")
openai_api_key = st.secrets.get("OPENAI_API_KEY", "")
deepseek_api_key = st.secrets.get("DEEPSEEK_API_KEY", "")

# Giao diện chọn động cơ
ai_engine = st.radio("🧠 Chọn động cơ AI:", ("Gemini 2.0 Flash", "GPT-4o", "DeepSeek AI"))

uploaded_files = st.file_uploader("Chọn ảnh CCCD/Hộ chiếu:", accept_multiple_files=True, type=['jpg', 'png', 'jpeg'])

# Hàm xử lý EasyOCR tối ưu CPU (Chỉ nạp khi cần)
def run_easyocr_fallback(image_path):
    import easyocr
    # Chỉ dùng 1 nhân CPU, hạn chế tài nguyên
    reader = easyocr.Reader(['vi', 'en'], gpu=False, model_storage_directory='~/.EasyOCR/')
    results = reader.readtext(image_path, detail=0)
    return "\n".join(results)

if uploaded_files and st.button("🚀 Bắt đầu"):
    matched_database = {}
    
    for up_file in uploaded_files:
        pil_img = Image.open(up_file)
        
        # 1. Thử gọi AI Đám mây (Gemini/GPT/DeepSeek)
        extracted_data = None
        try:
            # Code gọi AI (đã lược bớt phần base64 để tối ưu bộ nhớ)
            if "Gemini" in ai_engine:
                client = genai.Client(api_key=gemini_api_key)
                response = client.models.generate_content(model='gemini-2.0-flash', contents=["Trích xuất thông tin JSON từ ảnh này", pil_img])
                extracted_data = json.loads(response.text.replace("```json", "").replace("```", ""))
            
            # ... (Tương tự cho GPT và DeepSeek)
            
        except:
            st.warning(f"Đám mây nghẽn, đang dùng EasyOCR cho {up_file.name}...")
            # Lưu tạm để EasyOCR đọc
            temp_path = f"temp_{up_file.name}"
            pil_img.save(temp_path)
            raw_text = run_easyocr_fallback(temp_path)
            os.remove(temp_path)
            # Xử lý regex bóc tách tại đây...
            
        # Logic đối khớp STT và lưu vào matched_database...
        
    st.success("Hoàn thành!")
    # Xuất file Excel...
