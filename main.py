from fastapi import FastAPI, HTTPException  # FastAPI 웹 프레임워크와 예외 처리
from pydantic import BaseModel             # 요청 데이터 유효성 검사를 위한 데이터 모델
from dotenv import load_dotenv              # .env 파일 로드
from openai import OpenAI                   # OpenAI API 사용
import os                                   # 운영체제 환경 변수 접근
import json                                 # JSON 파싱 및 직렬화
import re                                   # 정규식 사용 (GPT 응답에서 JSON 추출용)

# FastAPI app 생성
app = FastAPI()

# .env 파일 로드 (API Key 등 환경 변수 설정)
load_dotenv()

# OpenAI 클라이언트 생성 (환경 변수에서 API Key 가져오기)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 요청 데이터 모델 정의
class DocText(BaseModel):
    text: str  # 사용자가 전달할 계약서 텍스트

# POST /text 엔드포인트 정의
@app.post("/text")
def analyze_text(text: DocText):

    # GPT에게 전달할 프롬프트 구성
    prompt = f"""
    아래 계약서 내용을 읽고 위험 조항을 반드시 JSON 배열 형태로만 분석해줘.
    추가 설명이나 텍스트 없이 JSON만 반환해야 합니다.

    JSON 필드:
    - type: 조항 종류
    - risk_level: LOW|MEDIUM|HIGH
    - excerpt: 문제되는 원문 일부
    - reason: 위험한 이유
    - suggested_fix: 수정 제안

    계약서 내용:
    {text.text}
    """

    try:
        # GPT API 호출
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # 사용할 GPT 모델
            messages=[
                {"role": "system", "content": "너는 한국 계약서 분석 전문가야. 반드시 JSON 형식만 출력해라."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0  # 결과 일관성을 위해 온도 0
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

        # 분석 결과 반환
        return {"analysis": result_json}

    except json.JSONDecodeError:
        # GPT 응답이 JSON 형식이 아닌 경우 예외 처리
        raise HTTPException(status_code=500, detail="GPT 응답이 올바른 JSON 형식이 아닙니다.")

    except Exception as e:
        # 그 외 모든 예외 처리
        raise HTTPException(status_code=500, detail=str(e))
