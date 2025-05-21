import cv2
import json
import re
import pytesseract
import numpy as np
import google.generativeai as genai
from collections import Counter
import os


GEMINI_API = os.getenv("GEMINI_KEY")

def preprocess_text(text):
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    return lines

def most_frequent_element(word_list):
    return Counter(word_list).most_common(1)[0][0]

def process_predictions(wp_preds):
    word_level_predictions = []
    current = []
    str_rep = ""

    for w, p in wp_preds:
        if w not in ['<s>', '</s>', '[CLS]']:
            str_rep += w.replace('▁', ' ')
            if '▁' in w:
                if current:
                    word_level_predictions.append(most_frequent_element(current))
                current = [p]
            else:
                current.append(p)

        if w == '</s>':
            word_level_predictions.append(most_frequent_element(current))
            return str_rep.strip(), word_level_predictions
    return str_rep.strip(), word_level_predictions

def convert_predictions_to_json(tokens, predictions):
    result = {}
    current_entity = None
    current_label = None
    other_tokens = []  # Danh sách lưu các token "O"

    for token, prediction in zip(tokens, predictions):
        label = prediction

        if label == "O":  # Nếu không có nhãn, thêm token vào "Other"
            other_tokens.append(token)
            current_entity = None
            current_label = None
            continue

        # Tách B- và I- để lấy tên nhãn chính
        prefix, entity = label.split("-", 1)

        if prefix == "B":  # Bắt đầu một thực thể mới
            if entity not in result:  # Tạo key mới nếu chưa có
                result[entity] = []

            # Kết thúc thực thể hiện tại và bắt đầu thực thể mới
            current_entity = token
            current_label = entity

        elif prefix == "I" and current_label == entity:  # Phần tiếp theo của thực thể
            current_entity += " " + token

        # Kết thúc thực thể và thêm vào kết quả
        if current_entity and (prefix == "B" or label == "O"):
            result[current_label].append(current_entity)
            current_entity = None
            current_label = None

    # Thêm thực thể cuối cùng nếu còn sót
    if current_entity and current_label:
        result[current_label].append(current_entity)

    # Xử lý các token "Other"
    if other_tokens:
        result["Other"] = " ".join(other_tokens)

    # Chuyển danh sách có 1 phần tử thành chuỗi
    for key in result:
        if isinstance(result[key], list) and len(result[key]) == 1:
            result[key] = result[key][0]

    return result

def group_by_label(text, labels):
    categories = {
        'Company': [],
        'Position': [],
        'Department': [],
        'Name': [],
        'Phone': [],
        'Email': [],
        'Address': [],
        'Other': []
    }

    current_category = None
    current_value = []

    for word, label in zip(text.split(), labels):
        # Xử lý nhãn B-
        if label.startswith('B-'):
            if current_category:
                # Thêm phần tử đã ghép thành công vào category
                if len(current_value) > 1:
                    categories[current_category].append(' '.join(current_value))
                else:
                    categories[current_category].append(current_value[0])
            # Cập nhật nhãn mới
            current_category = label[2:]  # Lấy tên category từ nhãn B- (loại bỏ 'B-')
            current_value = [word]
        # Xử lý nhãn I- nhưng không có B- trước đó cho cùng category
        elif label.startswith('I-') and not current_category == label[2:]:
            # Khi gặp I- mà không có B- của label đó, coi I đầu tiên như B
            if current_category:
                # Thêm phần tử đã ghép thành công vào category
                if len(current_value) > 1:
                    categories[current_category].append(' '.join(current_value))
                else:
                    categories[current_category].append(current_value[0])
            # Cập nhật nhãn mới
            current_category = label[2:]  # Loại bỏ 'I-' và lấy tên category
            current_value = [word]
        # Nếu gặp I- và đã có B- trước đó
        elif label.startswith('I-') and current_category == label[2:]:
            current_value.append(word)
        if label == 'O':
            categories['Other'].append(word)
        # Xử lý nhãn 'O', tức là không thuộc vào bất kỳ category nào

    # Thêm phần tử cuối cùng sau khi duyệt hết
    if current_category:
        if len(current_value) > 1:
            categories[current_category].append(' '.join(current_value))
        else:
            categories[current_category].append(current_value[0])

    # Chuyển sang JSON format
    # with open(r"output\output.json", "w", encoding="utf-8") as json_file:
    #     json.dump(categories, json_file, ensure_ascii=False, indent=4)
    return categories

def extract_json_from_markdown(markdown_text):
    code_block_match = re.search(r"```json\n(.*?)\n```", markdown_text, re.DOTALL)
    if code_block_match:
        json_string = code_block_match.group(1)
        try:
            return json.loads(json_string)
        except json.JSONDecodeError as e:
            return f"Error decoding JSON in code block: {e}"  # Return error string


    json_match = re.search(r"\{(.*?)\}", markdown_text, re.DOTALL) # Look for curly braces
    if json_match:
      json_string = json_match.group(0) # Include the curly braces
      try:
          return json.loads(json_string)
      except json.JSONDecodeError as e:
          return f"Error decoding JSON in text: {e}" # Return error string

    return {}

def arrange_text(ocr_results):
    # Sort by y-coordinate first, then by x-coordinate
    ocr_results.sort(key=lambda x: (np.mean([p[1] for p in x[0]]), np.mean([p[0] for p in x[0]])))
    
    lines = []
    current_line = []
    prev_y = None
    line_height_threshold = 10  # Adjust for spacing
    
    for box, text, conf in ocr_results:
        avg_y = np.mean([p[1] for p in box])
        
        if prev_y is None or abs(avg_y - prev_y) < line_height_threshold:
            current_line.append((box, text))
        else:
            lines.append(current_line)
            current_line = [(box, text)]
        
        prev_y = avg_y
    
    if current_line:
        lines.append(current_line)
    
    formatted_text = ""
    for line in lines:
        line.sort(key=lambda x: np.mean([p[0] for p in x[0]]))
        max_x = max([np.mean([p[0] for p in box]) for box, _ in line])
        min_x = min([np.mean([p[0] for p in box]) for box, _ in line])
        
        text_positions = {np.mean([p[0] for p in box]): text for box, text in line}
        sorted_positions = sorted(text_positions.keys())
        
        adjusted_line = ""
        last_x = min_x
        for x in sorted_positions:
            spaces = int((x - last_x) / 10)  # Adjust spacing factor
            adjusted_line += " " * spaces + text_positions[x]
            last_x = x + len(text_positions[x]) * 5  # Adjust spacing
        
        formatted_text += adjusted_line + "\n"
    
    return formatted_text.strip()

def normalize_text(text):
    cleaned_text = re.sub(r'\s+', ' ', text)
    return cleaned_text.strip()

def format_chat(chat_history, message):
    for item in chat_history:
        item["content"] = item.pop("parts")
    if item["role"] == "model":
        item["role"] = "assistant"
    
    chat_history.append({"role": "user", "content": message})
    return chat_history   

def detect_language(image=None, text='', llm=None):
    if image is not None:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Use pytesseract to extract text from the image
        text = pytesseract.image_to_string(image, lang='jpn+kor+vie')
    sample = """STAAS  & HALSEYu¿
지적 소유권 전문 ti #AL

스티븐 보그너

파트너

미국 워싱턴 D.C. 뉴욕 애버뉴 1201 번지

북서04.), 78 (우) 20005

전화 1.202.434.1800 / 팩스 +1.202.434.1801
0|8[9I: garner@s-n-h.com

'B|MO|E: www.staasandhalsey.com"""
    sample2 = """… 살이         ~    +»              aki      rh
が

wo        でMY

QkQ mnごS

のアーああっ   ^ /%

-%                 bé

^. 내 hai               1
1:        ~ 5 채써싸       으으

Ne @

FPTソフトウェ 株式会社

グエン・ヴァン・トゥアン
ディジタルイノベーションセンター
ブロックチェインソリューション Cog
리패륜

1

Mobile: +84 96 595 0707
Email: tuannv13@fsoft.com.vn

FPT Cau Giay Building, 17 Duy Tân Srr.

- Cau Giay Dist., Hanoi, Vietnam

Tel: +84 (24) 37689048

www.fpt-software.com"""
    chat_history = [
        {
            "role": "user",
            "parts": f"You are helpful system. Base on this text OCR from bussiness card below, respone language main in langugae card (include English, Japanese, Korean, Vietnamese). Unless English is full text, dont'n chose English. If it is none of the languages listed, then the response is 'unknown'.\n Text Ocr: \n`{sample}`. Only response is one lowercase word."
        },
        {
            "role": "model",
            "parts": "korean"
        },
        {
            "role": "user",
            "parts": f"You are helpful system. Base on this text OCR from bussiness card below, respone language main in langugae card (include English, Japanese, Korean, Vietnamese). Unless English is full text, dont'n chose English. If it is none of the languages listed, then the response is 'unknown'.\n Text Ocr: \n`{sample2}`. Only response is one lowercase word."
        },
        {
            "role": "model",
            "parts": "japanese"
        }
    ]
    
    message = f"Goodjob!. Continue. Base on this text OCR from bussiness card below, respone language main in langugae card (include English, Japanese, Korean, Vietnamese). Unless English is full text, dont'n chose English. If it is none of the languages listed, then the response is 'unknown'.\n Text Ocr: \n`{text}`. Only response is one lowercase word that is language you chose."
    if llm == None:
        genai.configure(api_key=GEMINI_API)
        # for model in genai.list_models():
            # print(model)
        # Create the model
        generation_config = {
        "temperature": 0.22,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 10,
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
        tmp = format_chat(chat_history, message)
        a = llm(tmp, max_new_tokens=500)
        response = a[0]["generated_text"][-1]["content"]
    
    return response.replace('\n', '').strip() if any(word in response for word in ["korean", "japanese", "vietnamese", "english"]) else "unknown"

        
    
 
# Example usage
if __name__ == "__main__":
    pass

