import streamlit as st
import google.generativeai as genai
from PIL import Image
import pandas as pd
import io
import json

st.set_page_config(page_title="Hệ thống Scan Giấy Tờ Gemini AI", page_icon="🪪", layout="wide")
st.title("🪪 Hệ thống Trích xuất Giấy tờ Hàng loạt bằng Gemini AI")
st.markdown("Ứng dụng sử dụng lõi công nghệ tân tiến nhất **Gemini 3.5 Flash**, tối ưu hóa tốc độ xử lý hàng loạt và độ chính xác.")

# CẤU HÌNH BẢO MẬT API KEY:
try:
    default_api_key = st.secrets["GEMINI_API_KEY"]
except:
    default_api_key = ""

# Ô nhập key dự phòng trực tiếp trên giao diện web
api_key = st.text_input("🔑 Google Gemini API Key (Đã tự động cấu hình hoặc nhập mới):", value=default_api_key, type="password")

# Lựa chọn loại giấy tờ cần quét cho đợt này
loai_giay_to = st.selectbox(
    "📁 Chọn loại giấy tờ bạn chuẩn bị tải lên hàng loạt:",
    ("Căn cước công dân / VNeID (Mặt trước hoặc mặt sau)", "Hộ chiếu (Passport - Việt Nam & Nước ngoài)")
)

# Giao diện tải nhiều tệp cùng lúc
uploaded_files = st.file_uploader(
    "📸 Chọn hoặc kéo thả cùng lúc nhiều ảnh giấy tờ vào đây:", 
    type=['jpg', 'png', 'jpeg'], 
    accept_multiple_files=True
)

if uploaded_files:
    st.info(f"🔗 Đã ghi nhận {len(uploaded_files)} tệp ảnh sẵn sàng để Gemini phân tích.")
    
    if st.button("🚀 Bắt đầu trích xuất bằng Gemini AI", type="primary"):
        if not api_key:
            st.warning("Vui lòng cấu hình hoặc nhập Gemini API Key để kích hoạt hệ thống!")
        else:
            # Kích hoạt kết nối đến siêu máy tính Google Gemini
            genai.configure(api_key=api_key)
            
            # ĐÃ CẬP NHẬT: Cấu hình mô hình thế hệ mới nhất Gemini 3.5 Flash
            model = genai.GenerativeModel('gemini-3.5-flash')
            
            all_results = [] # Mảng lưu dữ liệu tổng hợp để xuất Excel
            
            # Thanh tiến trình chạy trực quan
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, up_file in enumerate(uploaded_files):
                status_text.text(f"🤖 Gemini 3.5 Flash đang đọc và phân tích tệp ({idx + 1}/{len(uploaded_files)}): {up_file.name}")
                
                try:
                    # Mở ảnh trực tiếp bằng thư viện Pillow truyền thẳng cho Gemini
                    pil_img = Image.open(up_file)
                    
                    # Thiết lập câu lệnh Prompt chuyên nghiệp ép Gemini xuất cấu trúc định dạng JSON phẳng
                    if "Căn cước" in loai_giay_to:
                        prompt = """
                        Bạn là một hệ thống AI OCR chuyên nghiệp cấu trúc dữ liệu cho khách sạn.
                        Hãy đọc thật kỹ hình ảnh Căn cước công dân hoặc VNeID này (đây có thể là ảnh mặt trước hoặc ảnh mặt sau).
                        Hãy bỏ qua mọi hoa văn bảo mật, vệt lóa sáng của đèn, trích xuất thông tin chính xác và trả về DUY NHẤT một chuỗi định dạng JSON theo đúng cấu trúc dưới đây, không giải thích gì thêm:
                        {
                            "Số Định Danh / Hộ Chiếu": "Ghi số CCCD gồm 12 chữ số (nếu là mặt trước), nếu không thấy hoặc là mặt sau thì ghi Không tìm thấy",
                            "Họ và tên": "Ghi họ và tên bằng chữ IN HOA CÓ DẤU đầy đủ",
                            "Ngày tháng năm sinh": "Ghi định dạng DD/MM/YYYY, nếu không thấy thì ghi Không tìm thấy",
                            "Địa chỉ thường trú / Quốc tịch": "Ghi cụ thể địa chỉ nơi thường trú (nếu là mặt trước), nếu không thấy thì ghi Không tìm thấy"
                        }
                        """
                    else:
                        prompt = """
                        Bạn là một hệ thống AI OCR chuyên nghiệp cấu trúc dữ liệu cho khách sạn.
                        Hãy đọc thật kỹ hình ảnh Hộ chiếu (Passport) này (Hỗ trợ cả hộ chiếu Việt Nam và tất cả các quốc gia trên thế giới).
                        Hãy trích xuất thông tin chính xác và trả về DUY NHẤT một chuỗi định dạng JSON theo đúng cấu trúc dưới đây, không giải thích gì thêm:
                        {
                            "Số Định Danh / Hộ Chiếu": "Ghi mã số hộ chiếu (Passport No) bao gồm cả chữ và số",
                            "Họ và tên": "Ghi họ và tên bằng chữ IN HOA đầy đủ",
                            "Ngày tháng năm sinh": "Ghi định dạng ngày sinh DD/MM/YYYY",
                            "Địa chỉ thường trú / Quốc tịch": "Ghi rõ tên quốc gia / Quốc tịch (Nationality) bằng tiếng Việt hoặc tiếng Anh"
                        }
                        """
                    
                    # Gửi ảnh và lệnh cho Gemini AI xử lý
                    response = model.generate_content([prompt, pil_img])
                    
                    # Bóc tách dọn dẹp chuỗi JSON từ phản hồi văn bản của AI
                    raw_text = response.text.strip()
                    clean_json = raw_text.replace("```json", "").replace("```", "").strip()
                    
                    # Chuyển chuỗi JSON thành Dictionary Python
                    data_dict = json.loads(clean_json)
                    
                    # Đồng bộ dữ liệu xếp vào hàng dòng của bảng Excel
                    row_data = {
                        "STT": idx + 1,
                        "Tên File Ảnh": up_file.name,
                        "Loại Giấy Tờ": loai_giay_to,
                        "Số Định Danh / Hộ Chiếu": data_dict.get("Số Định Danh / Hộ Chiếu", "Không tìm thấy"),
                        "Họ và tên": data_dict.get("Họ và tên", "Không tìm thấy"),
                        "Ngày tháng năm sinh": data_dict.get("Ngày tháng năm sinh", "Không tìm thấy"),
                        "Địa chỉ thường trú / Quốc tịch": data_dict.get("Địa chỉ thường trú / Quốc tịch", "Không tìm thấy")
                    }
                    all_results.append(row_data)
                    
                except Exception as e:
                    st.error(f"Lỗi khi xử lý file {up_file.name}: {e}")
                    # Ghi nhận dòng lỗi để không làm lệch danh sách hàng Excel
                    all_results.append({
                        "STT": idx + 1, "Tên File Ảnh": up_file.name, "Loại Giấy Tờ": loai_giay_to,
                        "Số Định Danh / Hộ Chiếu": "Lỗi đọc ảnh", "Họ và tên": "Lỗi đọc ảnh",
                        "Ngày tháng năm sinh": "Lỗi đọc ảnh", "Địa chỉ thường trú / Quốc tịch": f"Chi tiết: {e}"
                    })
                
                # Cập nhật % thanh tiến trình chạy
                progress_bar.progress((idx + 1) / len(uploaded_files))
            
            status_text.text("✅ Đã hoàn thành trích xuất dữ liệu bằng trí tuệ nhân tạo Gemini 3.5 Flash!")
            
            # Hiển thị dữ liệu và đóng gói file Excel tải về
            if all_results:
                df = pd.DataFrame(all_results)
                st.subheader("📊 Bảng kết quả trích xuất tổng hợp dữ liệu")
                st.dataframe(df, use_container_width=True)
                
                # Tạo file Excel trực tiếp vào RAM server
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False, sheet_name='DuLieuKhachHang')
                
                st.markdown("---")
                st.subheader("📥 Tải về báo cáo file Excel dữ liệu")
                st.download_button(
                    label="📥 Bấm vào đây để tải file Excel (.xlsx)",
                    data=excel_buffer.getvalue(),
                    file_name="BaoCao_QuetGiayTo_GeminiAI.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type="primary"
                )
