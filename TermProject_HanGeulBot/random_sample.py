import json
import random
from hanspell import spell_checker
import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
import re

# ------------------------------------------------------------
# 이 스크립트는 두 가지 방법으로 "입력 텍스트 -> 맞춤법 교정된 출력 텍스트" 쌍을 생성하여
# 별도의 JSON 파일로 저장하는 역할을 합니다.
#
# 1. 국립국어원 제공 JSON 데이터셋 기반:
#    - 주어진 JSON 파일(MXEC2202210100.json)에서 원문과 교정문 쌍을 가져와 그 중 일부를 랜덤 샘플링합니다.
#    - 샘플링한 문장을 "맞춤법을 고쳐주세요:"라는 프롬프트 형태의 입력과, 정제된 교정문을 출력으로 지정합니다.
#
# 2. 랜덤 생성 문장 기반:
#    - GPT-2 기반 한국어 언어모델을 로드하여, 사전에 정의된 단어 리스트 중 랜덤한 단어를 골라 해당 단어를 기반으로 문장을 생성합니다.
#    - 생성된 문장에 hanspell 패키지를 사용하여 맞춤법 교정을 한 후,
#      "맞춤법을 고쳐주세요:"라는 프롬프트가 포함된 입력과 교정된 문장을 페어로 만듭니다.
#
# 두 경우 모두 결과를 JSON 파일(random_sample.json)에 덧붙이는 형태로 저장합니다.
# ------------------------------------------------------------

# 모델 및 토크나이저 경로
model_path = "./d0c0df48bf2b2c9350dd855021a5b216f560c0c7"
tokenizer_path = "./d0c0df48bf2b2c9350dd855021a5b216f560c0c7"

# 데이터셋 관련 파일 경로
output_file = "./dataset/random_sample.json"  # 생성한 데이터셋을 저장할 파일
input_file = "./dataset/MXEC2202210100.json" # 국립국어원 JSON 파일 경로

def load_model():
    """
    사전 학습된 GPT-2 언어 모델과 토크나이저를 로드하고 GPU 사용 가능 시 GPU로 이동합니다.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.to(device)
    return model, tokenizer, device

def generate_random_sentence(model, tokenizer, device, input_text):
    """
    입력 텍스트(input_text)를 기반으로 모델을 사용하여 문장을 생성합니다.
    생성 과정:
    - 최대 길이 50 토큰까지 문장 생성
    - 반복되는 n-gram 방지
    - 마침표('.')가 등장하면 그 지점에서 문장을 마칩니다.
    """
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
    output = model.generate(
        input_ids,
        max_length=50,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.encode('.')[0]
    )
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # 마침표가 등장하면 해당 마침표까지만 문장을 자릅니다.
    if '.' in generated_text:
        generated_text = generated_text.split('.')[0] + '.'

    return generated_text

def correct_spelling(input_text):
    """
    hanspell을 사용하여 입력 문장에 대한 맞춤법 교정을 수행하고, 교정된 문장을 반환합니다.
    """
    corrected = spell_checker.check(input_text)
    return corrected.checked

def clean_text(text):
    """
    텍스트 전처리 함수:
    - 특정 특수문자 제거 (.,)
    - &name\d+& 형태로 들어간 이름 태그 제거
    - 너무 짧은 문장은 빈 문자열로 처리
    """
    # 특수문자 제거
    text = re.sub(r"[.,]", "", text)
    # &name\d+& 형태 제거
    text = re.sub(r"\S*&name\d+&\S*", "", text)
    # 길이가 2 미만이면 빈 문자열 반환
    if len(text) < 2:
        return ""
    return text

def load_existing_data():
    """
    기존에 주어진 국립국어원 JSON 파일에서 원본 문장(original_form)과 교정된 문장(corrected_form)을 읽어옵니다.
    전처리를 통해 불필요한 부분을 제거하고, (원문, 교정문) 쌍을 리스트로 반환합니다.
    """
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print("파일을 찾을 수 없습니다.")
        return []

    utterances = []
    for document in dataset.get('document', []):
        for utterance in document.get('utterance', []):
            original_form = utterance.get('original_form', '')
            corrected_form = utterance.get('corrected_form', '')

            # 전처리 적용
            original_form = clean_text(original_form)
            corrected_form = clean_text(corrected_form)

            # 전처리 후 유효한 문장만 리스트에 추가
            if original_form and corrected_form:
                utterances.append((original_form, corrected_form))

    return utterances

def get_random_samples(utterances, num_samples=3):
    """
    주어진 (원문, 교정문) 쌍 리스트 중에서 num_samples 개수를 랜덤 샘플링합니다.
    샘플링된 결과를 '맞춤법을 고쳐주세요: 원문' 형태의 input_texts와 교정문 output_texts로 분리하여 반환합니다.
    """
    random.seed()  # 랜덤 시드를 고정하지 않아 매 실행마다 결과가 달라집니다.
    sampled_utterances = random.sample(utterances, num_samples)
    input_texts = [f"맞춤법을 고쳐주세요: {item[0]}" for item in sampled_utterances]
    output_texts = [item[1] for item in sampled_utterances]
    return input_texts, output_texts

def prepare_data_for_training(is_random):
    """
    랜덤 단어 기반으로 GPT-2를 통해 문장을 생성하고, hanspell로 맞춤법 교정을 수행합니다.
    - 사전에 정의된 random_words 리스트에서 단어를 하나 고른 뒤, 이를 기반으로 문장을 생성.
    - 생성된 문장을 맞춤법 교정 후, '맞춤법을 고쳐주세요: 생성문장' 형태의 입력과 교정문을 페어로 반환.
    """
    model, tokenizer, device = load_model()

    input_texts = []
    output_texts = []

    # 랜덤 문장 생성을 위한 단어 리스트 (사전 정의)
    random_words = [
        "사랑", "기후", "운동", "기술", "사회", "음악", "여행", "책", "날씨", "경제",
        "게임", "교육", "정치", "문화", "영화", "음식", "취미", "직업", "사회적", "기업",
        "철학", "사회적", "미래", "과학", "예술", "디지털", "로봇", "우주", "행복", "건강",
        "사회적", "보안", "패션", "자율성", "언어", "소셜", "개발", "탐험", "환경", "가치",
        "혁신", "인공지능", "정보", "리더십", "책임", "평등", "연대", "공정", "진보", "자유",
        "도전", "상상", "창의성", "협력", "연구", "테크", "디자인", "커리어", "창업", "복지",
        "디지털화", "로봇공학", "인터넷", "스마트폰", "기술적", "전략", "친환경", "소셜미디어",
        "브랜딩", "인테리어", "자율주행", "블록체인", "클라우드", "빅데이터", "AI", "교육과정",
        "프로그래밍", "데이터", "네트워크", "연구개발", "스타트업", "경제학", "금융", "자산관리",
        "비즈니스", "트렌드", "스마트시티", "디지털트윈", "모바일", "웨어러블", "카메라", "5G",
        "IoT", "스마트홈", "헬스케어", "글로벌", "사회적책임", "전자상거래", "디지털화", "e커머스"
    ]

    # 여기서는 5개의 문장을 생성해봄
    for i in range(5):
        # 랜덤 단어 선택
        random_word = random.choice(random_words)

        # 해당 단어를 시작점으로 문장 생성
        sentence = generate_random_sentence(model, tokenizer, device, random_word)
        # 맞춤법 교정
        corrected_sentence = correct_spelling(sentence)

        # "맞춤법을 고쳐주세요:" 라는 프롬프트 형태의 입력-출력 쌍 저장
        input_texts.append(f"맞춤법을 고쳐주세요: {sentence}")
        output_texts.append(corrected_sentence)

    return input_texts, output_texts

def append_to_json(input_texts, output_texts):
    """
    생성된 input_texts, output_texts 데이터를 기존 JSON 파일에 덧붙여 저장합니다.
    - 기존 random_sample.json 파일을 읽어 input_texts, output_texts를 확장한 뒤 다시 저장합니다.
    - 파일이 없을 경우 새로운 구조의 JSON을 생성합니다.
    """
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
    except FileNotFoundError:
        # 파일이 없으면 새로운 구조로 초기화
        existing_data = {"input_texts": [], "output_texts": []}

    # 새로운 데이터를 기존 데이터에 추가
    existing_data["input_texts"].extend(input_texts)
    existing_data["output_texts"].extend(output_texts)

    # 덧붙인 데이터를 다시 JSON 파일로 저장
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=4)

    print(f"랜덤 샘플링된 데이터가 {output_file}에 덧붙여졌습니다.")

def choose_learning_method():
    """
    사용자에게 데이터셋 생성 방식을 선택하게 한 뒤,
    선택된 방식에 따라 데이터 생성 로직을 실행합니다.
    1. 국립국어원 JSON 데이터 기반
    2. 랜덤 문장 생성 기반
    """
    print("데이터셋 생성 방법을 선택하세요:")
    print("1. 국립국어원 JSON 파일 기반으로 생성")
    print("2. 랜덤으로 문장 생성")

    choice = input("번호를 선택하세요: ")

    if choice == "1":
        # 기존 국립국어원 JSON 파일에서 (원문, 교정문) 쌍을 랜덤 추출
        print("기존 JSON 파일을 학습합니다.")
        utterances = load_existing_data()
        input_texts, output_texts = get_random_samples(utterances, num_samples=10)
        append_to_json(input_texts, output_texts)
    elif choice == "2":
        # GPT-2를 사용해 랜덤한 문장을 만들고, 맞춤법 교정한 뒤 JSON 파일에 저장
        print("랜덤 문장을 학습합니다.")
        input_texts, output_texts = prepare_data_for_training(is_random=True)
        append_to_json(input_texts, output_texts)

# 메인 실행부
if __name__ == "__main__":
    choose_learning_method()
