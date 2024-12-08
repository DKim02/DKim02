from flask import Flask, render_template, request, jsonify
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from hanspell import spell_checker  # Hanspell 사용

# ------------------------------------------------------------
# 이 Flask 애플리케이션은 다음 세 가지 맞춤법 검사 방식을 제공합니다:
# 1. 파인튜닝되지 않은 기본 모델(untuned_model)을 사용한 맞춤법 교정
# 2. 파인튜닝된 모델(fine_tuned_model)을 사용한 맞춤법 교정
# 3. Hanspell 패키지를 이용한 맞춤법 교정
#
# 사용자는 웹 인터페이스(index.html)에서 텍스트를 입력하고, 
# 원하는 검사기(untuned, tuned, hanspell)를 선택하여 맞춤법 교정을 요청할 수 있습니다.
#
# 주요 로직:
# - '/' 라우트: index.html 렌더링(메인 페이지)
# - '/check' 라우트: POST 방식으로 텍스트 및 검사기 타입을 받아 맞춤법 검사 결과 반환
#
# 이 코드는 Flask 앱을 실행시킨 뒤, localhost:5000 (또는 지정한 포트)에서 
# 웹페이지에 접속해 텍스트를 입력하고 검사 결과를 확인할 수 있습니다.
# ------------------------------------------------------------

app = Flask(__name__)

# 파인튜닝된 모델 및 토크나이저 로드 경로
model_path = './fine_tuned_model'
tokenizer_path = './fine_tuned_model'

# 파인튜닝되지 않은 모델 로드
# "j5ng/et5-typos-corrector"는 기본 한글 맞춤법 교정 T5 모델
untuned_model = T5ForConditionalGeneration.from_pretrained("j5ng/et5-typos-corrector")
untuned_tokenizer = T5Tokenizer.from_pretrained("j5ng/et5-typos-corrector")

# 파인튜닝된 모델 로드
fine_tuned_model = T5ForConditionalGeneration.from_pretrained(model_path)
fine_tuned_tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)

# GPU 사용 가능 여부 확인하여 모델 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
untuned_model.to(device)
fine_tuned_model.to(device)

@app.route('/')
def index():
    """
    메인 페이지 렌더링: index.html 반환
    사용자가 웹 UI를 통해 텍스트를 입력하고, 검사 방식을 선택할 수 있는 인터페이스 제공
    """
    return render_template('index.html')

@app.route('/check', methods=['POST'])
def check_spelling():
    """
    맞춤법 검사 요청을 처리하는 엔드포인트.
    사용자가 입력한 텍스트와 선택한 검사기 타입을 받아, 해당 방식으로 맞춤법 교정 후 결과를 JSON 형태로 반환.
    """
    text = request.form.get('text')  # 사용자가 입력한 텍스트
    checker_type = request.form.get('checker')  # 선택된 검사기 유형 (untuned, tuned, hanspell)

    if not text:
        return jsonify({'error': '텍스트를 입력해주세요.'})

    # 선택된 검사 방식에 따라 처리
    if checker_type == 'model_untuned':
        # 파인튜닝되지 않은 모델로 맞춤법 교정
        input_encoding = untuned_tokenizer("맞춤법을 고쳐주세요: " + text, return_tensors="pt").to(device)
        input_ids = input_encoding.input_ids
        attention_mask = input_encoding.attention_mask

        # T5 모델을 사용해 결과 생성
        output_encoding = untuned_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=128,
            num_beams=5,
            early_stopping=True,
        )
        output_text = untuned_tokenizer.decode(output_encoding[0], skip_special_tokens=True)

        return jsonify({
            'original_text': text,
            'checked_text': output_text
        })

    elif checker_type == 'model_tuned':
        # 파인튜닝된 모델로 맞춤법 교정
        input_encoding = fine_tuned_tokenizer("맞춤법을 고쳐주세요: " + text, return_tensors="pt").to(device)
        input_ids = input_encoding.input_ids
        attention_mask = input_encoding.attention_mask

        # 파인튜닝된 모델을 사용한 결과 생성
        output_encoding = fine_tuned_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=128,
            num_beams=5,
            early_stopping=True,
        )
        output_text = fine_tuned_tokenizer.decode(output_encoding[0], skip_special_tokens=True)

        return jsonify({
            'original_text': text,
            'checked_text': output_text
        })

    elif checker_type == 'hanspell':
        # Hanspell 라이브러리를 사용한 맞춤법 교정
        try:
            corrected_text = spell_checker.check(text).checked
            return jsonify({
                'original_text': text,
                'checked_text': corrected_text
            })
        except Exception as e:
            return jsonify({
                'error': f'Hanspell 처리 중 오류 발생: {str(e)}'
            })

    else:
        # 지원하지 않는 검사기 타입일 경우 에러 반환
        return jsonify({'error': '잘못된 검사기 선택입니다.'})

if __name__ == '__main__':
    # 개발용 서버 실행 (디버그 모드)
    # 실제 배포 시에는 WSGI 서버를 사용하여 실행 권장
    app.run(debug=True)
