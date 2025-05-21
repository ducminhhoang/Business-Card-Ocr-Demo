import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
import torch
from src.utils.utils import group_by_label, process_predictions, extract_json_from_markdown, normalize_text, arrange_text, format_chat
import os
import google.generativeai as genai
from paddleocr import PaddleOCR

GEMINI_API = os.getenv("GEMINI_KEY")


def correction(text_ocr, lang, model=None):
    sample1="""DAI HOCQUÖC GIAHANOI
VNU
DAI HOC QUOC GIAHA NOI                                                                                                                                        TRUONG DAI HOCCONGNGHE
PGS. TS. HOANG VAN XIEM
Chü nhiém, B mn Ky thuat Robot
DAIHQC
CONG NGHE
im
Nha E3,144 Xuan thüy,Cau Giay,Ha Ni
Dién thoai: 0378 608 113
etNam
Email: xiemhoang@vnu.edu.vn"""
    correct1 = """ĐẠI HỌC QUỐC GIA HÀ NỘI
VNU
ĐẠI HỌC QUỐC GIA HÀ NỘI                                                                                                                                        TRƯỜNG ĐẠI HỌC CÔNG NGHỆ
PGS. TS. HOÀNG VĂN XIÊM
Chủ nhiệm, Bộ môn Kỹ thuật Robot
ĐẠI HỌC
CÔNG NGHỆ
im
Nhà E3,144 Xuân Thủy,Cầu Giấy,Hà Nội
Điện thoại: 0378 608 113
Việt Nam
Email: xiemhoang@vnu.edu.vn"""
    sample2 = """Amkor
Technology
NGUYEN VAN TAM
ITManager
CONG TY TNHHAMKOR TECHNOLOGY VIET NAM
CHU
L só CN5B,KCN Yén Phong II-C,Thi tran Ch,Xa Tam Giang
Sales/Pl
XaDong Tién,Huyén Yen Phong,Tinh Bäc NinhViet Nam
Masó thué2301195652Sódién thoai:0916726338(VN
Tam.NguyenVan@amkor.com"""
    correct2 = """Amkor
Technology
NGUYỄN VĂN TÂM
IT Manager
CÔNG TY TNHH AMKOR TECHNOLOGY VIỆT NAM
CHU
Lô số CN5B,KCN Yên Phong II-C, Thị trấn Chờ, Xã Tam Giang
Sales/Pl
Xã Đông Tiến, Huyện Yên Phong, Tỉnh Bắc Ninh Việt Nam
Mã số thuế: 2301195652, Số điện thoại: 0916726338 (VN)
Tam.NguyenVan@amkor.com"""
    chat_history = [
        {
           "role": "user",
           "parts": [
               f"You are an helpful assistant in correcting and refining information text from business card ocr. Below is a text ocr from business card. \n\nOCR text:\n`{sample1}` Correct common OCR errors (e.g., misspellings, character substitutions) where possible. Correct according to Vietnamese characteristics. **Return the text in the following origin format.**\n\nOnly respone result, no comment."
           ]
        },
        {
            "role": "model",
            "parts": [
                correct1
            ]
        },
        {
           "role": "user",
           "parts": [
               f"You are an helpful assistant in correcting and refining information text from business card ocr. Below is a text ocr from business card. \n\nOCR text:\n`{sample2}` Correct common OCR errors (e.g., misspellings, character substitutions) where possible. Correct according to Vietnamese characteristics. **Return the text in the following origin format.**\n\nOnly respone result, no comment."
           ]
        },
        {
            "role": "model",
            "parts": [
                correct2
            ]
        }
    ]
    
    message = f"You are an helpful assistant in correcting and refining information text from business card ocr. Below is a text ocr from business card. \n\nOCR text:\n`{text_ocr}` Correct common OCR errors (e.g., misspellings, character substitutions) where possible. Correct according to {lang} characteristics. **Return the text in the following origin format.**\n\nOnly respone result, no comment."
    
    if model == None:
        
        genai.configure(api_key=GEMINI_API)

        generation_config = {
        "temperature": 0.22,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 500,
        "response_mime_type": "text/plain",
        }

        model = genai.GenerativeModel(
        model_name="gemma-3-12b-it",
        generation_config=generation_config,
        )

        chat_session = model.start_chat(
        history=chat_history
        )
        response = chat_session.send_message(message)
        response = response.text
    else:
        a = model(format_chat(chat_history, message), max_new_tokens=500)
        response = a[0]["generated_text"][-1]["content"]
    # print(response.text)
    
    return response
    

def infer(text, model, tokenizer, label_list, lang, device="cpu", llm=None):
    text_norm = normalize_text(text)
    tokenized_inputs = tokenizer(
        text_norm.split(),
        is_split_into_words=True,
        return_tensors="pt",  # Trả về tensor PyTorch
        truncation=True,            # Cắt chuỗi nếu quá dài
        # padding=True
        padding="max_length",
        max_length=96
    )
    tokenized_inputs = {key: value.to(device) for key, value in tokenized_inputs.items() if key != 'word_ids'}
    # Dự đoán
    with torch.no_grad():
        outputs = model(**tokenized_inputs)
    input_ids = tokenized_inputs["input_ids"].numpy()
    attention_mask = tokenized_inputs["attention_mask"].numpy()

    # Lấy nhãn dự đoán
    # logits = outputs.logits
    logits = torch.tensor(outputs[0])
    predictions = torch.argmax(logits, dim=-1).squeeze().tolist()
    
    # Mapping từ token ID -> nhãn trong label_list
    # word_ids = tokenized_inputs.word_ids()
    final_predictions = [
        label_list[pred]
        for pred in predictions
    ]

    tokens = tokenizer.convert_ids_to_tokens(tokenized_inputs['input_ids'].squeeze().tolist())
    wp_preds = list(zip(tokens, final_predictions))
    str_rep, word_level_predictions = process_predictions(wp_preds)
    # assert len(str_rep.strip().split(" ")) == len(word_level_predictions)
    c = group_by_label(str_rep, word_level_predictions)
    return c

def infer_img(img, model, tokenizer, label_list, lang, device="cpu", llm=None):
    dic = {
        'english': ['en'],
        'japanese': ['ja', 'en'],
        'korean': ['ko', 'en'],
        'vietnamese': ['vi', 'en']
        }
    dic_pp = {
        'english': 'en',
        'japanese': 'japan',
        'korean': 'korean',
        'vietnamese': 'vi'
    }
    # reader = easyocr.Reader(dic[lang])  # Nhận diện tiếng Anh, Nhật, Hàn, Trung Quốc
    
    # # Trích xuất thông tin
    # result = reader.readtext(img)
    
    ocr = PaddleOCR(use_angle_cls=True, lang=dic_pp[lang])
    result = ocr.ocr(img, cls=True)
    result = [(r[0], r[1][0], r[1][1]) for r in result[0]]
    text = arrange_text(result)
    print("================================")
    print(text)
    print("================================")
    text = correction(text, lang, llm)
    print(text)
    re = infer(text, model, tokenizer, label_list, lang, device, llm)
    return re
    