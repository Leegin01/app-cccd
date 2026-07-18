import streamlit as st
import requests

st.set_page_config(page_title="AI Scan Giấy Tờ", page_icon="🪪")
st.title(" Trích xuất thông tin CCCD & Hộ Chiếu")

# THAY CHUỖI API KEY CỦA EM VÀO ĐÂY
API_KEY = "47H23gkjeYnzE1xABP1AkxX3N3WyAE8y"

# Thanh lựa chọn loại giấy tờ cần quét
loai_giay_to = st.selectbox(
    "Bạn muốn quét loại giấy tờ nào?",
    ("Căn cước công dân / VNeID", "Hộ chiếu (Passport)")
)

st.markdown("Hướng dẫn: Tải ảnh rõ nét lên, hệ thống đám mây sẽ xử lý trong vài giây.")
up_file = st.file_uploader("Tải ảnh giấy tờ lên", type=['jpg', 'png', 'jpeg'])

if up_file:
    st.image(up_file, caption="Ảnh gốc đã tải lên", use_container_width=True)
    
    if st.button("Bắt đầu trích xuất", type="primary"):
        with st.spinner("Đang gửi ảnh lên hệ thống AI đám mây..."):
            try:
                # Cấu hình URL dựa trên loại giấy tờ người dùng chọn
                if loai_giay_to == "Căn cước công dân / VNeID":
                    url = "https://api.fpt.ai/vision/idr/vnm"
                else:
                    # Đường dẫn API chuyên đọc Hộ chiếu của FPT
                    url = "https://api.fpt.ai/vision/passport/vnm"
                
                headers = {"api-key": API_KEY}
                files = {"image": up_file.getvalue()}
                
                # Gửi yêu cầu lên FPT.AI
                response = requests.post(url, headers=headers, files=files)
                result = response.json()
                
                # Kiểm tra kết quả trả về từ máy chủ
                if response.status_code == 200 and result.get("errorCode") == 0:
                    data = result["data"][0]
                    st.success("Trích xuất dữ liệu thành công!")
                    
                    # Tạo bảng hiển thị linh hoạt theo từng loại giấy tờ
                    if loai_giay_to == "Căn cước công dân / VNeID":
                        info = {
                            "Số CCCD": data.get("id", "Không tìm thấy"),
                            "Họ và tên": data.get("name", "Không tìm thấy"),
                            "Ngày sinh": data.get("dob", "Không tìm thấy"),
                            "Nơi thường trú": data.get("address", "Không tìm thấy")
                        }
                    else:
                        # Các trường thông tin đặc trưng của Hộ chiếu
                        info = {
                            "Số Hộ Chiếu (Passport No)": data.get("passport_number", "Không tìm thấy"),
                            "Họ và tên": data.get("name", "Không tìm thấy"),
                            "Ngày sinh": data.get("dob", "Không tìm thấy"),
                            "Số định danh / Số CMND cũ": data.get("id_number", "Không tìm thấy"),
                            "Quốc tịch": data.get("nationality", "Không tìm thấy"),
                            "Ngày hết hạn": data.get("expiry_date", "Không tìm thấy")
                        }
                        
                    st.table(info)
                else:
                    st.error(f"Lỗi hệ thống AI nhận diện: {result.get('errorMessage', 'Không xác định')}")
                    
            except Exception as e:
                st.error(f"Đã xảy ra lỗi kết nối mạng: {e}")
