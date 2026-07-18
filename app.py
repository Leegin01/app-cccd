import streamlit as st
import google.generativeai as genai
from PIL import Image
import pandas as pd
import io
import json
from pyzbar.pyzbar import decode
import cv2
import numpy as np
import time  

st.set_page_config(page_title="Hệ thống Scan Giấy Tờ Cấu Trúc Cao", page_icon="🪪", layout="wide")
st.title("🪪 Hệ thống Trích xuất & So khớp CCCD Hai Mặt Tự Động")
st.markdown("Hệ thống thông minh kết hợp **Quét mã QR**, **Siêu AI Gemini 3.1 Pro (Phân tích chuyên sâu)** và thuật toán chống nghẽn mạng (Auto-Retry).")

# CẤU HÌNH BẢO MẬT API KEY (CHẠY NGẦM):
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except:
    api_key = ""

# Lựa chọn loại giấy tờ cần quét cho đợt này
loai_giay_to = st.selectbox(
    "📁 Chọn loại giấy tờ bạn chuẩn bị tải lên hàng loạt:",
    ("Căn cước công dân / VNeID (Quét hỗn hợp cả 2 mặt)", "Hộ chiếu (Passport - Việt Nam & Nước ngoài)")
)

# Giao diện tải nhiều tệp cùng lúc
uploaded_files = st.file_uploader(
    "📸 Chọn hoặc kéo thả cùng lúc nhiều ảnh giấy tờ (Thoải mái thả cả mặt trước và mặt sau):", 
    type=['jpg', 'png', 'jpeg'], 
    accept_multiple_files=True
)

if uploaded_files:
    st.info(f"🔗 Đã ghi nhận {len(uploaded_files)} tệp ảnh sẵn sàng xử lý.")
    
    if st.button("🚀 Bắt đầu trích xuất & So khớp dữ liệu", type="primary"):
        if not api_key:
            st.error("🚨 Không tìm thấy API Key! Vui lòng cấu hình GEMINI_API_KEY trong phần Secrets của Streamlit Cloud.")
        else:
            genai.configure(api_key=api_key)
            
            # ĐÃ CẬP NHẬT: Sử dụng mô hình chuyên sâu Gemini 3.1 Pro theo yêu cầu
            model = genai.GenerativeModel('gemini-3.1-pro')
            
            matched_database = {}
            unmatched_records = [] 
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, up_file in enumerate(uploaded_files):
                status_text.text(f"🔍 Gemini 3.1 Pro đang phân tích tệp ({idx + 1}/{len(uploaded_files)}): {up_file.name}")
                
                try:
                    pil_img = Image.open(up_file)
                    qr_extracted = False
                    extracted_data = {}
                    
                    # ---------------------------------------------------------
                    # THAO TÁC 1: THỬ QUÉT MÃ QR TRƯỚC (NHANH VÀ MIỄN PHÍ HOÀN TOÀN)
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
                                
                                # Chờ 2 giây để giao diện cập nhật cho đẹp
                                time.sleep(2)
                    
                    # ---------------------------------------------------------
                    # THAO TÁC 2: DÙNG GEMINI 3.1 PRO (BẢO VỆ BẰNG AUTO-RETRY)
                    # ---------------------------------------------------------
                    if not qr_extracted:
                        if "Căn cước" in loai_giay_to:
                            prompt = """
                            Bạn là một hệ thống AI OCR nghiệp vụ khách sạn cao cấp. Hãy phân tích hình ảnh Căn cước công dân/VNeID này.
                            Đây có thể là MẶT TRƯỚC hoặc MẶT SAU của thẻ.
                            NHIỆM VỤ QUAN TRỌNG: 
                            - Nếu là MẶT SAU, hãy nhìn vào hai dòng mã vạch ký tự MRZ ở dưới cùng (hoặc văn bản trên thẻ) để tìm ra dãy SỐ CCCD GỒM 12 CHỮ SỐ.
                            - Tìm ngày cấp (ngày/tháng/năm ở mặt sau) và đặc điểm nhân dạng.
                            Trả về kết quả DUY NHẤT dưới dạng một chuỗi JSON chuẩn định dạng sau, không giải thích gì thêm:
                            {
                                "Loại mặt": "Điền 'Mặt trước' hoặc 'Mặt sau'",
                                "Số Định Danh / Hộ Chiếu": "Điền 12 số CCCD tìm được (Kể cả tìm ở dòng MRZ mặt sau). Nếu không thấy ghi Không tìm thấy",
                                "Họ và tên": "Điền họ tên IN HOA CÓ DẤU (nếu là mặt trước), mặt sau ghi Không tìm thấy",
                                "Ngày tháng năm sinh": "Định dạng DD/MM/YYYY (nếu là mặt trước), mặt sau ghi Không tìm thấy",
                                "Địa chỉ thường trú / Quốc tịch": "Địa chỉ cụ thể (nếu là mặt trước), mặt sau ghi Không tìm thấy",
                                "Ngày cấp": "Định dạng DD/MM/YYYY (thường nằm ở mặt sau), mặt trước ghi Không tìm thấy",
                                "Đặc điểm nhân dạng": "Ghi cụ thể dòng đặc điểm nhân dạng ở mặt sau, mặt trước ghi Không tìm thấy"
                            }
                            """
                        else:
                            prompt = """
                            Hãy đọc hình ảnh Hộ chiếu này và trả về chuỗi JSON duy nhất:
                            {
                                "Loại mặt": "Hộ chiếu",
                                "Số Định Danh / Hộ Chiếu": "Mã số hộ chiếu",
                                "Họ và tên": "Họ và tên IN HOA",
                                "Ngày tháng năm sinh": "DD/MM/YYYY",
                                "Địa chỉ thường trú / Quốc tịch": "Tên quốc gia / Quốc tịch",
                                "Ngày cấp": "Ngày cấp hộ chiếu DD/MM/YYYY",
                                "Đặc điểm nhân dạng": "Không áp dụng"
                            }
                            """
                        
                        # VÒNG LẶP CHỐNG NGHẼN MẠNG DÀNH CHO DÒNG PRO (TỰ ĐỘNG CHỜ NẾU BỊ LỖI 429)
                        success = False
                        for attempt in range(3): # Cho phép thử tối đa 3 lần
                            try:
                                response = model.generate_content([prompt, pil_img])
                                clean_json = response.text.strip().replace("```json", "").replace("```", "").strip()
                                extracted_data = json.loads(clean_json)
                                success = True
                                break # Nếu API đọc thành công thì thoát vòng lặp Retry ngay
                            except Exception as api_err:
                                error_msg = str(api_err)
                                # Bắt đúng lỗi 429 hoặc Quota để yêu cầu máy chủ "thở"
                                if "429" in error_msg or "quota" in error_msg.lower():
                                    status_text.text(f"⏳ Mô hình Pro cần xử lý chậm lại! Đang tự động chờ 45 giây làm mát API (Lần thử {attempt+1}/3)...")
                                    time.sleep(45) # Ép máy chủ ngủ 45 giây rồi mới chạy tiếp
                                else:
                                    raise api_err # Nếu là lỗi khác thì ném ra ngoài để báo lỗi
                        
                        if not success:
                            raise Exception("Đã hết số lần thử lại tự động do máy chủ Google quá tải.")
                        
                        # Giãn cách 3 giây giữa mỗi bức ảnh để API không bị dồn dập
                        time.sleep(3)

                    # ---------------------------------------------------------
                    # THAO TÁC 3: THUẬT TOÁN SO KHỚP & ĐỒNG BỘ HAI MẶT
                    # ---------------------------------------------------------
                    id_key = extracted_data.get("Số Định Danh / Hộ Chiếu", "Không tìm thấy")
                    
                    if id_key == "Không tìm thấy" or len(id_key) < 5:
                        extracted_data["Tên File Gốc"] = up_file.name
                        unmatched_records.append(extracted_data)
                    else:
                        if id_key in matched_database:
                            current_record = matched_database[id_key]
                            current_record["Tên File Mặt Sau"] = up_file.name if extracted_data.get("Loại mặt") == "Mặt sau" else current_record.get("Tên File Mặt Sau", "Chưa quét")
                            if extracted_data.get("Loại mặt") == "Mặt trước":
                                current_record["Tên File Mặt Trước"] = up_file.name
                            
                            for field in ["Họ và tên", "Ngày tháng năm sinh", "Địa chỉ thường trú / Quốc tịch", "Ngày cấp", "Đặc điểm nhân dạng"]:
                                if extracted_data.get(field) != "Không tìm thấy" and extracted_data.get(field) != "":
                                    current_record[field] = extracted_data.get(field)
                                    
                            current_record["Trạng Thái So Khớp"] = "✅ Khớp Thành Công 2 Mặt"
                        else:
                            matched_database[id_key] = {
                                "Số Định Danh / Hộ Chiếu": id_key,
                                "Tên File Mặt Trước": up_file.name if extracted_data.get("Loại mặt") == "Mặt trước" else "Chưa quét mặt trước",
                                "Tên File Mặt Sau": up_file.name if extracted_data.get("Loại mặt") == "Mặt sau" else "Chưa quét mặt sau",
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
            
            status_text.text("✅ Đã hoàn thành phân tích toàn bộ dữ liệu bằng Gemini 3.1 Pro!")
            
            # ---------------------------------------------------------
            # KHU VỰC 3: DỰNG BẢNG & ĐÓNG GÓI EXCEL CHUẨN CÔNG NGHIỆP
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
                    "Tên File Mặt Sau": "Lỗi đọc dữ liệu hình ảnh",
                    "Họ và tên": err_data.get("Họ và tên", "Mờ chữ"),
                    "Ngày tháng năm sinh": err_data.get("Ngày tháng năm sinh", "-"),
                    "Địa chỉ thường trú / Quốc tịch": err_data.get("Địa thường trú / Quốc tịch", "-"),
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
                
                st.subheader("📊 Bảng kết quả cấu trúc và đối sánh song biên")
                st.dataframe(df, use_container_width=True)
                
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False, sheet_name='LưuTrúKháchHàng')
                
                st.markdown("---")
                st.subheader("📥 Tải về báo cáo tổng hợp")
                st.download_button(
                    label="📥 Bấm vào đây để tải file Excel đối sánh (.xlsx)",
                    data=excel_buffer.getvalue(),
                    file_name="BaoCao_SoKhop_CCCD_GeminiPro.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type="primary"
                )
