import streamlit as st
import requests

st.set_page_config(page_title="AI Scan CCCD", page_icon="🪪")
st.title("🪪 Trích xuất thông tin CCCD/VNeID")

# THAY CHUỖI API KEY CỦA EM VÀO ĐÂY
API_KEY = "47H23gkjeYnzE1xABP1AkxX3N3WyAE8y"

st.markdown("Hướng dẫn: Tải ảnh thẻ lên, hệ thống đám mây sẽ xử lý trong vài giây.")
up_file = st.file_uploader("Tải ảnh thẻ lên", type=['jpg', 'png', 'jpeg'])

if up_file:
    st.image(up_file, caption="Ảnh gốc", use_container_width=True)
    
    if st.button("Bắt đầu trích xuất", type="primary"):
        with st.spinner("Đang gửi ảnh lên hệ thống AI đám mây..."):
            try:
                # Cấu hình giao tiếp với FPT AI API
                url = "https://api.fpt.ai/vision/idr/vnm"
                headers = {"api-key": API_KEY}
                files = {"image": up_file.getvalue()}
                
                # Gửi yêu cầu đi
                response = requests.post(url, headers=headers, files=files)
                result = response.json()
                
                # Kiểm tra kết quả trả về
                if response.status_code == 200 and result.get("errorCode") == 0:
                    data = result["data"][0]
                    st.success("Trích xuất thành công!")
                    
                    # Rút trích và hiển thị dữ liệu
                    info = {
                        "Số CCCD": data.get("id", "Không tìm thấy"),
                        "Họ và tên": data.get("name", "Không tìm thấy"),
                        "Ngày sinh": data.get("dob", "Không tìm thấy"),
                        "Nơi thường trú": data.get("address", "Không tìm thấy")
                    }
                    st.table(info)
                else:
                    st.error(f"Lỗi hệ thống AI nhận diện: {result.get('errorMessage', 'Không xác định')}")
                    
            except Exception as e:
                st.error(f"Đã xảy ra lỗi kết nối mạng: {e}")
