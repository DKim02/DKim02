# 맞춤법 검사 ㄱ?(Ver. KR)

## 1. 프로젝트 개요
본 프로젝트는 한국어 맞춤법 교정 모델을 학습시키고, 이를 활용한 웹 애플리케이션을 개발하여 사용자가 실시간으로 맞춤법을 교정할 수 있도록 하는 시스템입니다. 사용자가 입력한 텍스트에서 맞춤법 오류를 자동으로 찾아 교정하고, 이를 웹 인터페이스를 통해 즉시 확인할 수 있는 서비스를 제공합니다. 이 시스템은 Flask를 사용한 웹 서버와 PyTorch를 이용한 맞춤법 교정 모델을 결합하여 성능을 향상시켰습니다.
이 프로젝트는 한국어 맞춤법 교정의 정확도를 높이고, 사용자가 쉽게 접근할 수 있는 실용적인 도구를 제공하는 것을 목표로 하고 있습니다. 이를 통해, 텍스트 작성에서의 오류를 최소화하고, 더 신뢰할 수 있는 문서 작성이 가능해지도록 돕습니다.


## 2. 프로젝트 배경 및 필요성
한국어는 다양한 맞춤법 규정과 예외적인 규칙들로 인해 많은 사람들이 맞춤법 실수를 자주 범하는 언어입니다. 이를 해결하기 위한 자동화된 맞춤법 교정 시스템은 교육, 비즈니스, 연구 등 여러 분야에서 유용하게 활용될 수 있습니다. 하지만 기존의 맞춤법 검사기는 대부분 문어체에 집중되어 있으며, 구어체에 대한 교정은 부족한 실정입니다. 본 프로젝트는 이러한 한계를 극복하고, 구어체 교정까지 가능하도록 시스템을 확장하는 것을 목표로 합니다.

1. 기존 맞춤법 교정기의 한계
•	기존 맞춤법 검사기들은 문어체 위주로 개발되었고, 구어체나 고유명사, 특수한 문법에 대해서는 정확한 교정이 이루어지지 않습니다.
•	머신러닝 기반의 검사기들은 여전히 정확도나 완전성에서 부족하며, 비즈니스, 교육, 연구 등의 중요한 분야에서 신뢰도가 떨어질 수 있습니다.

2. 사용자 참여형 데이터셋 개선
•	본 프로젝트는 집단지성의 접근 방식을 활용하여 교정기가 제대로 교정하지 못하는 오류를 데이터셋에 추가하거나 랜덤 샘플을 생성하여 모델 성능을 강화하는 방법을 도입합니다.
•	교정기 성능 향상을 위해 사용자 피드백을 반영하고, 다양한 문법적 오류를 교정할 수 있도록 데이터셋을 확장합니다.

3. 구어체 교정의 중요성
•	기존 맞춤법 검사기는 대부분 문어체에 집중되어 있어 구어체 교정에 한계가 있습니다.
•	구어체의 교정 또한 중요한 요소로, 사용자들이 실생활에서 쓰는 문장에 대한 교정을 제공함으로써, 더 실용적인 맞춤법 검사기를 목표로 합니다.


## 3. 시스템 설계
본 시스템은 데이터셋 생성, 모델 훈련과 웹 애플리케이션 세 가지 주요 구성 요소로 나뉩니다. 각 구성 요소는 맞춤법 교정 기능을 제공하기 위해 긴밀하게 연결되어 있습니다.
1. 데이터셋 생성
2. 모델 훈련(및파인튜닝)
3. 웹 애플리케이션

파일구조는 [깃허브](https://github.com/DKim02/DKim02/tree/main/KoreanSpellCheck)를 참고해주시길 바랍니다.

## 3.1 데이터셋 생성
본 시스템에서 사용되는 데이터셋은 세 가지 방법으로 생성됩니다.

### 1. 랜덤 데이터셋 생성(1) ("random_sample.py" 코드 실행)

코드 실행 후 1번을 선택하면, KoGPT2 모델을 사용하여 무작위 문장을 생성하고, 이를 hanspell을 통해 교정한 후, 추가적인 전처리 과정을 거쳐 random_sample.json 파일로 저장됩니다. 

#### 1.1 KoGPT2로 무작위 문장 생성 
#### 1.2 생성된 문장을 Hanspell로 문장 교정
#### 1.3 전처리 및 random_sample.json 파일로 저장

이 데이터셋은 모델 훈련에 사용되며, 주로 문어체에서의 다양한 맞춤법 오류와 교정된 문장을 포함하고 있습니다.

### 2. 랜덤 데이터셋 생성(2) ("random_sample.py" 코드 실행)

코드 실행 후 2번을 선택하면, 국립 국어원의 "맞춤법 교정 말뭉치 2022" 데이터셋을 사용하여 주로 구어체 문장을 중심으로 데이터셋을 생성합니다. 

#### 2.1 국립 국어원에서 제공하는 말뭉치 사용 (MXEC2202210100.json)
#### 2.2 형식에 맞게 해당 json파일을 불러와 전처리 한 후 random_sample.json 파일로 저장

이 데이터셋은 모델 훈련에 사용되며, 주로 구어체에서의 다양한 맞춤법 오류와 교정된 문장을 포함하고 있습니다.

### 3. 직접 형식에 맞춰서 입력

사용자가 맞춤법 검사기를 사용하면서, 개선이 되었으면 하는 부분을 직접 데이터셋에 입력하고, 파인튜닝시킬 수 있습니다.

#### 3.1 random_sample.json파일을 실행
#### 3.2 "input_texts"와 "output_texts"에 형식에 맞게 직접 기입 후 저장

<random_sample.json의 형식>

		{
   		 "input_texts": [
   		     "맞춤법을 고쳐주세요: 학우 여러분 남은 기말고사 끝까지 화이팅 해서 유종애미를 거둬주세요 ㅎㅎ"
 
 		   ],
  		  "output_texts": [
 		       "학우 여러분 남은 기말고사 끝까지 파이팅 해서 유종의 미를 거둬주세요 ㅎㅎ"
 		   ]
		}

추가로 공개된 데이터셋을 사용하고 싶다면 random_sample.py에서 파일을 입력 부분을 고치면 된다.
만약 "수정 전", "수정 후"의 쌍을 이루고 있지 않는 데이터 셋이라면 "수정 전"을 hanspell을 이용해 "수정 후"의 데이터 셋으로 만들 수 있다.

## 3.2 모델 훈련(및 파인튜닝)
본 프로젝트에서는 ET5 모델을 사용하여 맞춤법 교정 모델을 훈련시킵니다. 하지만 시간과 컴퓨터 성능의 한계로 인해, ET5 모델을 처음부터 훈련시키기보다는 et5-typos-corrector 모델을 사용하여 훈련을 진행했습니다. 이 모델은 이미 맞춤법 교정에 특화된 데이터셋으로 파인튜닝된 상태로 제공되며, 모델을 불러와 random_sample.json 데이터셋을 사용하여 추가적인 파인튜닝을 수행했습니다.

[et5-typos-corrector](https://huggingface.co/j5ng/et5-typos-corrector)
해당 링크를 통해 자세한 설명을 보실 수 있습니다. (PLM모델인 ET5를 사용하여 국립국어원이 제공하는 데이터셋을 학습하였습니다.)

파인튜닝 과정은 다음과 같이 진행되었습니다:

### 1. 기존 모델 로드: HuggingFace에서 제공하는 et5-typos-corrector 모델을 불러와 사용.
### 2. 데이터셋 로딩: 앞서 생성한 random_sample.json 파일을 로드하여 훈련 데이터로 활용.
### 3. 파인튜닝: 기존 모델에 새로운 데이터셋을 추가 학습하여 성능을 개선.
### 4. 모델 저장: 파인튜닝이 완료된 모델은 fine_tuned_model 폴더에 저장되어, 후속 작업에 사용됩니다.

이 과정에 대한 자세한 내용은 train_model.py 코드에서 확인할 수 있습니다.

## 3.3 웹 애플리케이션
웹 애플리케이션은 Flask 프레임워크를 사용하여 구현되었습니다. 사용자는 웹 페이지에서 텍스트를 입력하고, 제출 버튼을 클릭하면 실시간으로 맞춤법 교정을 받을 수 있습니다. 이 웹 애플리케이션은 사용자 친화적인 UI를 제공하며, 맞춤법 교정 기능을 빠르고 효율적으로 수행할 수 있도록 설계되었습니다.

웹 애플리케이션의 주요 구성 요소:

Flask 웹 서버:
app.py 파일에서 Flask 웹 서버를 실행합니다. ("requirement.txt"를 다운 받고 실행하셔야합니다.)
서버가 실행이 된 후 [링크](http://127.0.0.1:5000/)를 통해서 확인하실 수 있습니다.
웹 서버는 사용자가 입력한 텍스트를 받아, 해당 텍스트에 대한 맞춤법 교정을 수행한 후 교정된 텍스트를 반환합니다.
서버는 클라이언트와의 통신을 관리하며, 맞춤법 교정 모델을 통해 결과를 전달합니다.

*서버를 만들어서 배포하면 재정 이슈가 발생하여, 로컬로 만든 점 양해 부탁드립니다.

UI 설계:
index.html: 사용자에게 텍스트 입력란과 교정 결과를 출력하는 HTML 페이지를 제공합니다. 텍스트 입력란에 문장을 입력하고 "맞춤법 검사하기" 버튼을 클릭하면 서버로 요청을 보냅니다.

동작 흐름:
사용자가 웹 페이지에서 텍스트를 입력한 후, "맞춤법 검사하기" 버튼을 클릭하면 입력된 텍스트가 Flask 서버로 전송됩니다.
서버는 전송된 텍스트를 모델에 전달하여 맞춤법 교정을 수행합니다.
교정된 결과를 다시 웹 페이지로 반환하여 사용자가 교정된 텍스트를 확인할 수 있습니다.

웹 애플리케이션은 빠르고 직관적인 교정 결과를 제공하며, 다양한 입력 문장을 처리할 수 있습니다. 이를 통해 사용자는 실시간으로 맞춤법을 교정받을 수 있습니다.

## 4. 사용된 주요 라이브러리 및 도구
- Python: 프로젝트의 주 프로그래밍 언어
- Hugging Face: 사전 학습된 모델과 훈련 지원
- KoGPT2: 한국어 GPT-2 모델을 기반으로 한 문장 생성
- Flask: 웹 서버 구축을 위한 Python 프레임워크
- PyTorch: 머신러닝 모델 훈련과 추론을 위한 라이브러리
- ET5: 텍스트 수정 및 맞춤법 교정 모델 (Seq2seq방식의 모델)
- hanspell: 외부 맞춤법 검사기 모듈
- JSON: 데이터셋 저장 및 불러오기

## 5. 실험 및 결과
훈련된 모델을 사용하여 여러 가지 문장을 입력하고 맞춤법 교정 결과를 평가했습니다. 모델은 대부분의 일반적인 맞춤법 오류를 정확하게 교정했습니다. 그러나 고유명사나 특수한 문법 규칙에 대해서는 정확도가 다소 떨어졌습니다. 이를 개선하기 위한 추가적인 훈련이 필요합니다.

## 6. 문제점 및 개선 사항
모델이 고유명사나 특수한 문장 구조에서 정확하지 않은 교정 결과를 반환할 때가 있었습니다. 이를 개선하기 위해 다양한 데이터셋을 추가하여 훈련시키고, 파인튜닝을 통해 모델 성능을 더욱 향상시킬 계획입니다.

## 7. 결론
본 프로젝트는 한국어 맞춤법 교정 모델을 훈련시키고, 이를 웹 애플리케이션으로 구현하는 데 성공했습니다. 이 시스템은 사용자가 텍스트를 입력하고 즉시 교정된 결과를 확인할 수 있도록 해주며, 향후 개선 작업을 통해 더욱 정확한 맞춤법 교정 모델을 제공할 수 있습니다.

## 8. 참고문헌
- Hugging Face KoGPT2: https://huggingface.co/skt/kogpt2-base-v2
- ET5 모델: https://aiopen.etri.re.kr/et5Model
- 파인튜닝된 ET5 모델: https://huggingface.co/j5ng/et5-typos-corrector

## 9. 사용된 라이선스 (링크를 통해 라이선스를 확인하실 수 있습니다.)

1. **KoGPT2**:
   - **라이선스**: [CC-BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
   - **내용**: KoGPT2 모델은 CC-BY-NC-SA 4.0 라이선스 하에 공개되어 있습니다. 이 라이선스는 비상업적 사용만을 허용하며, 수정 및 재배포가 가능합니다. 상업적 용도로 사용 시 별도의 동의가 필요합니다.

2. **ET5 모델**:
   - **라이선스**: [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0)
   - **내용**: ET5 모델의 연구용 활용은 Apache 2.0 라이선스를 따릅니다. 이 라이선스는 상업적 사용, 수정 및 재배포를 허용합니다. 다만, 소스 코드 변경 사항을 명시하고 원 저작권 고지를 해야 하며, 특허권도 보호받습니다.

3. **py-hanspell**:
   - **라이선스**: [MIT License](https://opensource.org/licenses/MIT)
   - **내용**: py-hanspell은 MIT 라이선스 하에 제공됩니다. 이 라이선스는 상업적 사용, 수정 및 재배포를 자유롭게 허용하며, 소스 코드 변경 사항을 명시할 필요는 없지만, 원 저작자와 라이선스를 반드시 명시해야 합니다.

4. **Flask**:
   - **라이선스**: [BSD-3-Clause License](https://opensource.org/licenses/BSD-3-Clause)
   - **내용**: Flask는 BSD 3-Clause License 하에 제공됩니다. 이 라이선스는 상업적 사용 및 수정 후 재배포가 가능하지만, 소스 코드 수정 시 라이선스와 저작권 고지를 명시해야 합니다.

5. **PyTorch**:
   - **라이선스**: [BSD-3-Clause License](https://opensource.org/licenses/BSD-3-Clause)
   - **내용**: PyTorch는 BSD 3-Clause License 하에 제공됩니다. 상업적 사용과 수정 후 재배포가 가능하지만, 소스 코드 수정 시 라이선스와 저작권 고지를 명시해야 합니다.

6. **transformers (HuggingFace)**:
   - **라이선스**: [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0)
   - **내용**: HuggingFace Transformers 라이브러리는 Apache 2.0 라이선스 하에 제공됩니다. 소스 코드 수정 및 재배포가 자유로우며, 변경 사항을 명시하고 원 저작권 고지를 포함해야 합니다.

7. **sentencepiece**:
   - **라이선스**: [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0)
   - **내용**: SentencePiece는 Apache 2.0 라이선스 하에 제공됩니다. 사용 및 배포가 자유로우며, 라이선스를 명시해야 합니다.

8. **scikit-learn**:
   - **라이선스**: [BSD-3-Clause License](https://opensource.org/licenses/BSD-3-Clause)
   - **내용**: scikit-learn은 BSD 3-Clause License 하에 제공됩니다. 상업적 사용 및 수정 후 재배포가 가능합니다. 다만, 소스 코드 수정 시 라이선스와 저작권 고지를 명시해야 합니다.

9. **datasets (Hugging Face Datasets 라이브러리)**:
   - **라이선스**: [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0)
   - **내용**: Datasets 라이브러리는 Apache 2.0 라이선스 하에 제공되며, 상업적 사용과 수정 후 재배포가 가능합니다.
	