import streamlit as st
from google import genai
from openai import OpenAI
from PIL import Image
import pandas as pd
import io
import json
from pyzbar.pyzbar import decode
import cv2
import numpy as np
import time
import re
import tempfile
import os
import base64

st.set_page_config(page_title="Hệ thống Scan Giấy Tờ Tối Ưu Pro", page_icon="🪪", layout="wide")
st.title("🪪 Hệ thống Trích xuất & So khớp CCCD Tự Động")
st.markdown("Kiến trúc Đa Lõi (Bản tối ưu CPU): **QR Code** ➔ **[Gemini 3.5 Flash / GPT-4o / DeepSeek]** ➔ Dự phòng **EasyOCR (Lazy Load)**.")

# CẤU HÌNH BẢO MẬT API KEY TỪ KÉT SẮT SECRETS RUNTIME:
try:
    gemini_api_key = st.secrets.get("GEMINI_API_KEY", "")
    openai_api_key = st.secrets.get("OPENAI_API_KEY", "")
    deepseek_api_key = st.secrets.get("DEEPSEEK_API_KEY", "")
except:
    gemini_api_key = ""
    openai_api_key = ""
    deepseek_api_key = ""

# Thiết kế khu vực tùy chọn Động cơ AI và Loại giấy tờ cần quét
col1, col2 = st.columns(2)
with col1:
    ai_engine = st.radio("🧠 Chọn động cơ Trí tuệ nhân tạo (AI Engine):", 
                         ("Google Gemini 3.5 Flash (Siêu tốc)", "OpenAI GPT-4o (Cần có số dư API)", "DeepSeek AI (Tối ưu chi phí)"))
with col2:
    loai_giay_to = st.selectbox("📁 Chọn loại giấy tờ nghiệp vụ lễ tân:",
                                ("Căn cước công dân / VNeID (Quét hỗn hợp 2 mặt)", "Hộ chiếu (Passport - Việt Nam & Nước ngoài)"))

uploaded_files = st.file_uploader(
    "📸 Chọn hoặc kéo thả cùng lúc nhiều ảnh giấy tờ (Thoải mái thả hỗn hợp cả mặt trước và mặt sau):", 
    type=['jpg', 'png', 'jpeg'], 
    accept_multiple_files=True
)

# ==========================================
# KHU VỰC 1: CÁC HÀM XỬ LÝ ẢNH & ĐỊNH DẠNG DỮ LIỆU
# ==========================================
def encode_image_to_base64(image_file):
    """Mã hóa ảnh cấu trúc dữ liệu sang dạng Base64 phục vụ cho luồng Vision API"""
    return base64.b64encode(image_file.getvalue()).decode('utf-8')

# Hàm xử lý EasyOCR tối ưu CPU (Chỉ nạp khi thực sự cần thiết)
def run_easyocr_fallback(image_path):
    import easyocr
    reader = easyocr.Reader(['vi', 'en'], gpu=False)
    results = reader.readtext(image_path, detail=0)
    return "\n".join(results)

# ==========================================
# KHU VỰC 2: TIẾN TRÌNH VÒNG LẶP VÀ CƠ CHẾ CHUYỂN MẠCH AI
# ==========================================
if uploaded_files:
    st.info(f"🔗 Hệ thống ghi nhận {len(uploaded_files)} tệp ảnh sẵn sàng đưa vào tiến trình.")
    
    if st.button("🚀 Bắt đầu trích xuất & So khớp dữ liệu", type="primary"):
        # Kiểm tra điều kiện API Key
        if "Gemini" in ai_engine and not gemini_api_key:
            st.error("🚨 Không tìm thấy cấu hình Gemini API Key trong Secrets!")
            st.stop()
        elif "GPT-4o" in ai_engine and not openai_api_key:
            st.error("🚨 Không tìm thấy cấu hình OpenAI API Key trong Secrets!")
            st.stop()
        elif "DeepSeek" in ai_engine and not deepseek_api_key:
            st.error("🚨 Không tìm thấy cấu hình DeepSeek API Key trong Secrets!")
            st.stop()
            
        # Khởi tạo client kết nối (Cập nhật chuẩn SDK mới của Google)
        if "Gemini" in ai_engine:
            client_gemini = genai.Client(api_key=gemini_api_key)
        elif "GPT-4o" in ai_engine:
            client_openai = OpenAI(api_key=openai_api_key)
        else:
            client_deepseek = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")
            
        matched_database = {}
        unmatched_records = [] 
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, up_file in enumerate(uploaded_files):
            status_text.text(f"🔍 Đang tiến hành phân tích tệp ({idx + 1}/{len(uploaded_files)}): {up_file.name}")
            
            try:
                pil_img = Image.open(up_file)
                qr_extracted = False
                extracted_data = {}
                
                # ---------------------------------------------------------
                # LỚP PHÒNG THỦ 1: KIỂM TRA MÃ QR CODE
                # ---------------------------------------------------------
                if "Căn cước" in loai_giay_to:
                    decoded_objects = decode(pil_img)
                    if decoded_objects:
                        qr_data = decoded_objects[0].data.decode('utf-8')
                        parts = qr_data.split('|')
                        if len(parts) >= 6:
                            extracted_data = {
                                "Loại mặt": "Mặt trước",
                                "Số Định Danh / Hộ Chiếu": parts[0],
                                "Họ và tên": parts[2].upper(),
                                "Ngày tháng năm sinh": f"{parts[3][:2]}/{parts[3][2:4]}/{parts[3][4:]}" if len(parts[3]) == 8 else parts[3],
                                "Địa chỉ thường trú / Quốc tịch": parts[5],
                                "Ngày cấp": parts[6] if len(parts) > 6 else "Không tìm thấy",
                                "Đặc điểm nhân dạng": "Nằm ở mặt sau"
                            }
                            qr_extracted = True
                            time.sleep(0.1)
                
                # ---------------------------------------------------------
                # LỚP PHÒNG THỦ 2: SỬ DỤNG SIÊU AI ĐÁM MÂY
                # ---------------------------------------------------------
                if not qr_extracted:
                    prompt = """
                    Bạn là một AI OCR nghiệp vụ. Hãy phân tích hình ảnh ảnh giấy tờ (CCCD/Hộ chiếu) này.
                    NẾU LÀ MẶT SAU CCCD, nhiệm vụ tối quan trọng là tìm dãy SỐ CCCD CÓ 12 CHỮ SỐ ở dải ký tự mã vạch MRZ hoặc văn bản.
                    Trả về DUY NHẤT một chuỗi định dạng JSON theo cấu trúc phẳng dưới đây, tuyệt đối không giải thích hay thêm ký tự thừa:
                    {
                        "Loại mặt": "Điền cụ thể 'Mặt trước' hoặc 'Mặt sau' hoặc 'Hộ chiếu'",
                        "Số Định Danh / Hộ Chiếu": "Ghi cụ thể 12 số CCCD hoặc Số Hộ Chiếu tìm được",
                        "Họ và tên": "Ghi họ tên viết bằng CHỮ IN HOA CÓ DẤU đầy đủ",
                        "Ngày tháng năm sinh": "Định dạng ngày sinh DD/MM/YYYY",
                        "Địa chỉ thường trú / Quốc tịch": "Địa chỉ nơi thường trú hoặc tên Quốc tịch",
                        "Ngày cấp": "Định dạng ngày cấp giấy tờ DD/MM/YYYY",
                        "Đặc điểm nhân dạng": "Ghi cụ thể dòng đặc điểm nhân dạng"
                    }
                    """
                    
                    use_easyocr_fallback = False
                    
                    try:
                        if "Gemini" in ai_engine:
                            # CẬP NHẬT: Gọi trực tiếp mô hình Gemini 3.5 Flash siêu tốc
                            response = client_gemini.models.generate_content(
                                model='gemini-3.5-flash', 
                                contents=[prompt, pil_img]
                            )
                            clean_json = response.text.strip().replace("```json", "").replace("```", "").strip()
                            extracted_data = json.loads(clean_json)
                        elif "GPT-4o" in ai_engine:
                            base64_image = encode_image_to_base64(up_file)
                            response = client_openai.chat.completions.create(
                                model="gpt-4o",
                                messages=[
                                    {"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]}
                                ],
                                response_format={ "type": "json_object" }
                            )
                            clean_json = response.choices[0].message.content.strip()
                            extracted_data = json.loads(clean_json)
                        else:
                            base64_image = encode_image_to_base64(up_file)
                            response = client_deepseek.chat.completions.create(
                                model="deepseek-chat",
                                messages=[
                                    {"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]}
                                ],
                                response_format={ "type": "json_object" }
                            )
                            clean_json = response.choices[0].message.content.strip()
                            extracted_data = json.loads(clean_json)
                            
                    except Exception as api_err:
                        status_text.text(f"⚠️ {ai_engine} gặp sự cố kết nối. Đang nạp EasyOCR dự phòng...")
                        use_easyocr_fallback = True
                    
                    # ---------------------------------------------------------
                    # LỚP PHÒNG THỦ 3: EASYOCR (LAZY LOAD - TỐI ƯU CPU)
                    # ---------------------------------------------------------
                    if use_easyocr_fallback:
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as f:
                            f.write(up_file.getbuffer())
                            temp_path = f.name
                        
                        raw_text = run_easyocr_fallback(temp_path)
                        
                        extracted_data = {
                            "Loại mặt": "Không xác định (EasyOCR)",
                            "Số Định Danh / Hộ Chiếu": "Không tìm thấy",
                            "Họ và tên": "Không tìm thấy",
                            "Ngày tháng năm sinh": "Không tìm thấy",
                            "Địa chỉ thường trú / Quốc tịch": "Không tìm thấy",
                            "Ngày cấp": "Không tìm thấy",
                            "Đặc điểm nhân dạng": "Trích xuất cục bộ"
                        }
                        
                        if "Căn cước" in loai_giay_to:
                            clean_text_no_space = raw_text.replace(" ", "")
                            id_matches = re.findall(r"\d{12}", clean_text_no_space)
                            if id_matches: extracted_data["Số Định Danh / Hộ Chiếu"] = id_matches[0]
                            dob_matches = re.findall(r"\d{2}[/-]\d{2}[/-]\d{4}", raw_text)
                            if dob_matches: extracted_data["Ngày tháng năm sinh"] = dob_matches[0]
                            name_matches = re.findall(r"\b[A-ZÀ-Ỹ]{2,}(?:\s+[A-ZÀ-Ỹ]{2,})+\b", raw_text)
                            if name_matches:
                                system_words = ["CỘNG HÒA", "XÃ HỘI", "CHỦ NGHĨA", "VIỆT NAM", "CĂN CƯỚC", "CÔNG DÂN", "CỤC TRƯỞNG"]
                                filtered_names = [n for n in name_matches if n not in system_words]
                                if filtered_names: extracted_data["Họ và tên"] = filtered_names[0]
                        else:
                            clean_text_passport = raw_text.replace(" ", "").upper()
                            passport_matches = re.findall(r"[A-Z]\d{7}", clean_text_passport)
                            if passport_matches: extracted_data["Số Định Danh / Hộ Chiếu"] = passport_matches[0]
                            dob_matches = re.findall(r"\d{2}[/-]\d{2}[/-]\d{4}", raw_text)
                            if dob_matches: extracted_data["Ngày tháng năm sinh"] = dob_matches[0]
                                
                        os.remove(temp_path)

                # ---------------------------------------------------------
                # THAO TÁC 3: THUẬT TOÁN ĐỐI CHIẾU SONG BIÊN VÀ MERGE HỒ SƠ
                # ---------------------------------------------------------
                id_key = extracted_data.get("Số Định Danh / Hộ Chiếu", "Không tìm thấy")
                
                if id_key == "Không tìm thấy" or len(id_key) < 5:
                    extracted_data["Tên File Gốc"] = up_file.name
                    unmatched_records.append(extracted_data)
                else:
                    if id_key in matched_database:
                        current_record = matched_database[id_key]
                        current_record["Tên File Mặt Sau"] = up_file.name if "Mặt sau" in extracted_data.get("Loại mặt", "") else current_record.get("Tên File Mặt Sau", "Chưa quét")
                        if "Mặt trước" in extracted_data.get("Loại mặt", ""):
                            current_record["Tên File Mặt Trước"] = up_file.name
                        
                        for field in ["Họ và tên", "Ngày tháng năm sinh", "Địa chỉ thường trú / Quốc tịch", "Ngày cấp", "Đặc điểm nhân dạng"]:
                            if extracted_data.get(field) != "Không tìm thấy" and extracted_data.get(field) != "":
                                current_record[field] = extracted_data.get(field)
                                
                        current_record["Trạng Thái So Khớp"] = "✅ Khớp Thành Công 2 Mặt"
                    else:
                        matched_database[id_key] = {
                            "Số Định Danh / Hộ Chiếu": id_key,
                            "Tên File Mặt Trước": up_file.name if "Mặt trước" in extracted_data.get("Loại mặt", "") else "Chưa quét mặt trước",
                            "Tên File Mặt Sau": up_file.name if "Mặt sau" in extracted_data.get("Loại mặt", "") else "Chưa quét mặt sau",
                            "Họ và tên": extracted_data.get("Họ và tên", "Không tìm thấy"),
                            "Ngày tháng năm sinh": extracted_data.get("Ngày tháng năm sinh", "Không tìm thấy"),
                            "Địa chỉ thường trú / Quốc tịch": extracted_data.get("Địa chỉ thường trú / Quốc tịch", "Không tìm thấy"),
                            "Ngày cấp": extracted_data.get("Ngày cấp", "Không tìm thấy"),
                            "Đặc điểm nhân dạng": extracted_data.get("Đặc điểm nhân dạng", "Không tìm thấy"),
                            "Trạng Thái So Khớp": "⚠️ Chỉ mới quét 1 mặt"
                        }
                        
            except Exception as e:
                st.error(f"Lỗi hệ thống khi xử lý tệp {up_file.name}: {e}")
            
            progress_bar.progress((idx + 1) / len(uploaded_files))
        
        status_text.text(f"✅ Đã hoàn thành quy trình trích xuất dữ liệu bằng {ai_engine}!")
        
        # ---------------------------------------------------------
        # KHU VỰC 3: DỰNG BẢNG VÀ ĐÓNG GÓI EXCEL REPORT
        # ---------------------------------------------------------
        final_rows = []
        stt = 1
        
        for key, data in matched_database.items():
            data["STT"] = stt
            final_rows.append(data)
            stt += 1
            
        for err_data in unmatched_records:
            final_rows.append({
                "STT": stt,
                "Số Định Danh / Hộ Chiếu": "Lỗi/Không tìm thấy ID",
                "Tên File Mặt Trước": err_data.get("Tên File Gốc", "Lỗi"),
                "Tên File Mặt Sau": "Lỗi dữ liệu định dạng hình ảnh",
                "Họ và tên": err_data.get("Họ và tên", "Mờ nét"),
                "Ngày tháng năm sinh": err_data.get("Ngày tháng năm sinh", "-"),
                "Địa chỉ thường trú / Quốc tịch": err_data.get("Địa chỉ thường trú / Quốc tịch", "-"),
                "Ngày cấp": err_data.get("Ngày cấp", "-"),
                "Đặc điểm nhân dạng": err_data.get("Đặc điểm nhân dạng", "-"),
                "Trạng Thái So Khớp": "❌ Lỗi trích xuất"
            })
            stt += 1
            
        if final_rows:
            df = pd.DataFrame(final_rows)
            column_order = ["STT", "Số Định Danh / Hộ Chiếu", "Họ và tên", "Ngày tháng năm sinh", 
                            "Địa chỉ thường trú / Quốc tịch", "Ngày cấp", "Đặc điểm nhân dạng", 
                            "Tên File Mặt Trước", "Tên File Mặt Sau", "Trạng Thái So Khớp"]
            df = df[column_order]
            
            st.subheader("📊 Bảng kết quả cấu trúc và đối sánh tổng hợp")
            st.dataframe(df, use_container_width=True)
            
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='LưuTrúKháchHàng')
            
            st.markdown("---")
            st.subheader("📥 Xuất dữ liệu báo cáo")
            st.download_button(
                label="📥 Bấm vào đây để tải file Excel đối sánh (.xlsx)",
                data=excel_buffer.getvalue(),
                file_name="BaoCao_SoKhop_ToiUu.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary"
            )
