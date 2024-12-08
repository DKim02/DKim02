import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import Trainer, TrainingArguments
import json
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------
# 이 스크립트는 T5 모델을 활용하여 맞춤법 교정 모델을 파인튜닝하는 과정의 예시를 보여줍니다.
#
# 주요 흐름:
# 1. 사전 학습된 T5 맞춤법 교정 모델과 토크나이저 로드
# 2. 미리 준비된 JSON 데이터에서 "input_texts"와 "output_texts"를 불러옴
# 3. 데이터 전처리 후 훈련/검증 데이터로 분할
# 4. 토크나이저를 사용하여 텍스트를 인덱싱
# 5. PyTorch Dataset 형태로 변환
# 6. Trainer를 활용하여 파인튜닝 수행
# 7. 파인튜닝 완료 후 모델과 토크나이저 저장
#
# 주석을 통해 각 단계별로 역할을 명확히 설명하였습니다.
# ------------------------------------------------------------

# T5 모델 및 토크나이저 로드
# "j5ng/et5-typos-corrector" 모델은 미리 한글 맞춤법 교정에 최적화된 T5 계열 모델
model = T5ForConditionalGeneration.from_pretrained("j5ng/et5-typos-corrector")
tokenizer = T5Tokenizer.from_pretrained("j5ng/et5-typos-corrector")

# JSON 데이터 파일 로드
# 이 파일에는 {"input_texts": [...], "output_texts": [...]} 구조로 데이터가 저장되어 있음
input_file = "./dataset/random_sample.json"
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# "input_texts"와 "output_texts" 분리
input_texts = data["input_texts"]
output_texts = data["output_texts"]

# 훈련 세트와 검증 세트로 분할
# test_size=0.2는 20% 데이터를 검증용으로 사용
train_df, val_df = train_test_split(list(zip(input_texts, output_texts)), test_size=0.2, random_state=42)

# T5 모델 훈련에 맞추어 입력 문장 전처리
# "맞춤법을 고쳐주세요: " 문구를 앞에 붙여, 모델이 교정 작업을 해야 함을 명시
# 출력 문장 끝에 "."를 붙여 마침표로 문장 종결 형식 유지 (선택 사항)
train_input = ["맞춤법을 고쳐주세요: " + item[0] for item in train_df]
train_output = [item[1] + "." for item in train_df]

val_input = ["맞춤법을 고쳐주세요: " + item[0] for item in val_df]
val_output = [item[1] + "." for item in val_df]

# 토크나이징:
# max_length=128: 문장 최대 길이를 128 토큰으로 제한
# padding=True, truncation=True: 필요한 경우 패딩 및 잘라내기
train_encodings = tokenizer(train_input, max_length=128, padding=True, truncation=True)
train_labels_encodings = tokenizer(train_output, max_length=128, padding=True, truncation=True)

val_encodings = tokenizer(val_input, max_length=128, padding=True, truncation=True)
val_labels_encodings = tokenizer(val_output, max_length=128, padding=True, truncation=True)

# PyTorch Dataset 클래스 정의
# 모델 학습에 필요한 형태로 데이터를 만들어줌
class SpellCorrectionDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels_encodings):
        self.encodings = encodings
        self.labels_encodings = labels_encodings

    def __getitem__(self, idx):
        # encodings에서 인덱스 idx의 토큰 텐서들을 가져옴
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # labels는 labels_encodings의 input_ids를 사용
        item["labels"] = torch.tensor(self.labels_encodings["input_ids"][idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])

# 훈련/검증용 Dataset 생성
train_dataset = SpellCorrectionDataset(train_encodings, train_labels_encodings)
val_dataset = SpellCorrectionDataset(val_encodings, val_labels_encodings)

# TrainingArguments 설정
# 모델 출력 경로, 학습률, 배치 사이즈, 에폭 수, weight decay 등 설정
training_args = TrainingArguments(
    output_dir="./outputs",
    evaluation_strategy="epoch",   # 매 epoch마다 검증
    learning_rate=1e-4,
    per_device_train_batch_size=32,
    num_train_epochs=8,
    weight_decay=0.01,
    save_strategy="epoch",         # 매 epoch마다 체크포인트 저장
    metric_for_best_model="eval_loss", 
    greater_is_better=False        # eval_loss는 작을수록 좋음
)

# Trainer 초기화
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# 모델 파인튜닝 시작
trainer.train()

# 파인튜닝 완료 후 모델과 토크나이저 저장
model.save_pretrained("./fine_tuned_model", safe_serialization=False)
tokenizer.save_pretrained("./fine_tuned_model")

# PyTorch 형식으로 모델 가중치 저장
torch.save(model.state_dict(), './fine_tuned_model/pytorch_model.bin')

print("훈련 완료 및 모델 저장 완료")
