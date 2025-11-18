# Colab용 LangChain + OpenAI 테스트 코드
from openai import OpenAI
from dotenv import load_dotenv
from test_api import contract_text3
import os
import json
import re

# .env 파일 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# OpenAI 클라이언트 생성
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# GPT에게 전달할 프롬프트 구성
prompt = f"""
아래 계약서 내용을 읽고 위험 조항을 반드시 JSON 배열 형태로만 분석해줘.
각 조항에 대해 동일한 reason이 반복되지 않도록 해라.
추가 설명이나 텍스트 없이 JSON만 반환해야 합니다.

JSON 필드:
- type: 조항 종류
- risk_level: LOW|MEDIUM|HIGH
- excerpt: 문제되는 원문 일부
- reason: 위험한 이유
- suggested_fix: 수정 제안

계약서 내용:
{contract_text3}
"""

try:
    # GPT API 호출
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # 사용할 GPT 모델
        messages=[
            {"role": "system", "content": "너는 한국 계약서 분석 전문가야. 반드시 JSON 형식만 출력해라."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0  # 결과 일관성을 위해 0으로 설정
    )

    # GPT 응답에서 텍스트 추출
    result_str = response.choices[0].message.content

    # JSON 블록만 추출 (```json ... ``` 형태 가능 대비)
    match = re.search(r"```json\s*(.*?)```", result_str, re.DOTALL)
    if match:
        result_str = match.group(1)
    
    # 문자열 앞뒤 공백 제거
    result_str = result_str.strip()

    # JSON 변환
    result_json = json.loads(result_str)

    # 결과 출력
    print(json.dumps(result_json, indent=2, ensure_ascii=False))

except json.JSONDecodeError:
    print("GPT 응답이 올바른 JSON 형식이 아닙니다.")
except Exception as e:
    print("오류 발생:", str(e))
