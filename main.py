from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
import os
import json
import re  # ⬅ 수정됨, 정규식 사용

# FastAPI app
app = FastAPI()

# .env 로드
load_dotenv()

# OpenAI 클라이언트 생성
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 요청 데이터 모델
class DocText(BaseModel):
    text: str

@app.post("/text")
def analyze_text(text: DocText):

    # GPT에게 전달할 prompt 구성
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
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "너는 한국 계약서 분석 전문가야. 반드시 JSON 형식만 출력해라."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )

        # GPT 응답에서 텍스트 추출
        result_str = response.choices[0].message.content

        #JSON 블록만 추출 (```json ... ``` 형태 가능 대비)
        match = re.search(r"```json\s*(.*?)```", result_str, re.DOTALL)
        if match:
            result_str = match.group(1)
        
        # 문자열 정리
        result_str = result_str.strip()

        # JSON 변환
        result_json = json.loads(result_str)

        return {"analysis": result_json}

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="GPT 응답이 올바른 JSON 형식이 아닙니다.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
