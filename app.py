import streamlit as st
import cv2
import numpy as np
import re
import pytesseract
from PIL import Image
import tempfile
import os
import pandas as pd
import io

st.set_page_config(page_title="Hệ thống Quét Giấy Tờ Hàng Loạt", page_icon="🪪", layout="wide")
st.title("🪪 Hệ thống Trích xuất Giấy tờ hàng loạt & Xuất Excel")
st.markdown("Hệ thống sử dụng **AI Tesseract LSTM**, hỗ trợ tải lên nhiều tệp ảnh CCCD (mặt trước/mặt sau) hoặc Hộ chiếu cùng lúc.")

# Giao diện chọn loại giấy tờ chung cho đợt quét
loai_giay_to = st.selectbox(
    "📁 Chọn loại giấy tờ bạn chuẩn bị tải lên hàng loạt:",
    ("Căn cước công dân / VNeID", "Hộ chiếu (Passport)")
)

# ==========================================
# KHU VỰC 1: XỬ LÝ ẢNH BẰNG OPENCV (GIỮ NGUYÊN)
# ==========================================
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1], rect[3] = pts[np.argmin(diff)], pts[np.argmax(diff)]
    return rect

def perspective_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    width = max(int(np.linalg.norm(br - bl)), int(np.linalg.norm(tr - tl)))
    height = max(int(np.linalg.norm(tr - br)), int(np.linalg.norm(tl - bl)))
    dst = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (width, height))

def crop_and_align_card(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    ratio = img.shape[0] / 500.0
    orig = img.copy()
    img_resized = cv2.resize(img, (int(img.shape[1]/ratio), 500))
    
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)
    
    cnts, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    
    for c in cnts:
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) == 4:
            return perspective_transform(orig, approx.reshape(4, 2) * ratio)
    return orig 

def enhance_image_for_tesseract(cv_img):
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    thresh = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return thresh

# ==========================================
# KHU VỰC 2: GIAO DIỆN TẢI NHIỀU FILE & XỬ LÝ LẶP
# ==========================================
# Bật tính năng accept_multiple_files=True để chọn được nhiều ảnh cùng lúc
uploaded_files = st.file_uploader(
    "📸 Chọn hoặc kéo thả nhiều ảnh giấy tờ vào đây:", 
    type=['jpg', 'png', 'jpeg'], 
    accept_multiple_files=True
)

if uploaded_files:
    st.info(f"🔗 Đã ghi nhận {len(uploaded_files)} tệp ảnh sẵn sàng xử lý.")
    
    if st.button("🚀 Bắt đầu trích xuất hàng loạt", type="primary"):
        all_results = [] # Mảng chứa dữ liệu tổng hợp của tất cả các file
        
        # Tạo thanh tiến trình chạy trực quan trên giao diện web
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, up_file in enumerate(uploaded_files):
            # Cập nhật trạng thái xử lý từng file một
            status_text.text(f"🔄 Đang xử lý tệp ({idx + 1}/{len(uploaded_files)}): {up_file.name}")
            
            try:
                # 1. Lưu file tạm xuống ổ đĩa của server Cloud
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as f:
                    f.write(up_file.getbuffer())
                    temp_path = f.name
                
                # 2. Chạy OpenCV cắt phẳng ảnh và lọc nhị phân xóa nền
                aligned_img = crop_and_align_card(temp_path)
                if aligned_img is None:
                    os.remove(temp_path)
                    continue
                    
                enhanced_gray_img = enhance_image_for_tesseract(aligned_img)
                pil_img = Image.fromarray(enhanced_gray_img)
                
                # 3. Gọi lõi AI Tesseract LSTM song ngữ
                custom_config = r'--oem 1 --psm 3'
                raw_text = pytesseract.image_to_string(pil_img, lang='vie+eng', config=custom_config)
                
                # 4. Phân loại cấu trúc dữ liệu đồng nhất để xuất dòng Excel công nghiệp
                row_data = {
                    "STT": idx + 1,
                    "Tên File Ảnh": up_file.name,
                    "Loại Giấy Tờ": loai_giay_to,
                    "Số Định Danh / Hộ Chiếu": "Không tìm thấy",
                    "Họ và tên": "Không tìm thấy",
                    "Ngày tháng năm sinh": "Không tìm thấy",
                    "Địa chỉ thường trú / Quốc tịch": "Không tìm thấy"
                }
                
                # Áp dụng bộ lọc Regex trích xuất dữ liệu cũ của em
                if loai_giay_to == "Căn cước công dân / VNeID":
                    id_match = re.search(r"\d{12}", raw_text)
                    if id_match: row_data["Số Định Danh / Hộ Chiếu"] = id_match.group()

                    dob_match = re.search(r"\d{2}/\d{2}/\d{4}", raw_text)
                    if dob_match: row_data["Ngày tháng năm sinh"] = dob_match.group()

                    name_match = re.search(r"(?:Họ và tên|name)[:\s]*([A-ZÀ-Ỹ\s]+)", raw_text, re.IGNORECASE)
                    if name_match: row_data["Họ và tên"] = name_match.group(1).strip()

                    addr_match = re.search(r"(?:thường trú|residence)[:\s]*([^\n]+)", raw_text, re.IGNORECASE)
                    if addr_match: row_data["Địa chỉ thường trú / Quốc tịch"] = addr_match.group(1).strip()
                else:
                    # Cấu trúc Hộ chiếu
                    passport_no_match = re.search(r"[A-Z]\d{7}", raw_text)
                    if passport_no_match: row_data["Số Định Danh / Hộ Chiếu"] = passport_no_match.group()
                    
                    dob_match = re.search(r"\d{2}/\d{2}/\d{4}", raw_text)
                    if dob_match: row_data["Ngày tháng năm sinh"] = dob_match.group()
                    
                    name_match = re.search(r"(?:Họ và tên|name|Họ\/Surname)[:\s]*([A-ZÀ-Ỹ\s]+)", raw_text, re.IGNORECASE)
                    if name_match: row_data["Họ và tên"] = name_match.group(1).strip()
                    
                    nat_match = re.search(r"(?:Quốc tịch|Nationality)[:\s]*([A-ZÀ-Ỹ\s]+)", raw_text, re.IGNORECASE)
                    if nat_match: row_data["Địa chỉ thường trú / Quốc tịch"] = nat_match.group(1).strip()

                # Gom dữ liệu tệp này vào danh sách tổng
                all_results.append(row_data)
                
                # Xóa file rác tạm thời ngay sau khi quét xong để giải phóng Ram bộ nhớ
                os.remove(temp_path)
                
            except Exception as e:
                st.error(f"Lỗi khi xử lý file {up_file.name}: {e}")
            
            # Cập nhật thanh hiển thị % tiến trình cho người dùng theo dõi
            progress_bar.progress((idx + 1) / len(uploaded_files))
        
        status_text.text("✅ Đã hoàn thành xử lý toàn bộ danh sách tệp ảnh!")
        
        # ==========================================
        # KHU VỰC 3: HIỂN THỊ BẢNG & ĐÓNG GÓI FILE EXCEL
        # ==========================================
        if all_results:
            # Chuyển đổi danh sách kết quả thành một bảng dữ liệu Pandas DataFrame cấu trúc phẳng
            df = pd.DataFrame(all_results)
            
            st.subheader("📊 Bảng kết quả trích xuất tổng hợp")
            st.dataframe(df, use_container_width=True) # Hiển thị bảng tương tác cực đẹp mắt trên giao diện web
            
            # Kỹ thuật xuất file Excel trực tiếp vào luồng bộ nhớ đệm (BytesIO buffer) không tạo file rác vật lý
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                # Xuất bảng dữ liệu ra trang tính đầu tiên
                df.to_excel(writer, index=False, sheet_name='DanhSachKhachHang')
            
            st.markdown("---")
            st.subheader("📥 Tải về báo cáo Excel kết quả")
            
            # Tạo nút bấm tải xuống file chuẩn định dạng Excel .xlsx chuyên nghiệp
            st.download_button(
                label="📥 Bấm vào đây để tải file Excel (.xlsx)",
                data=excel_buffer.getvalue(),
                file_name="BaoCao_TrichXuat_GiayTo.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary"
            )
        else:
            st.warning("Không có dữ liệu nào được trích xuất thành công.")
