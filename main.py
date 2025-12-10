from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
import os

app = FastAPI()

# ----- AI 컬리지 지침 (여기에 네가 뽑아둔 헌법을 통째로 넣으면 됨) -----
AI_COLLEGE_SYSTEM_PROMPT = """
너는 'AI 컬리지 GPT OS'다.

여기에 AI 컬리지 지침/헌법 전체를 그대로 복붙하면 됨.
예: 출력 형식, 금지 규칙, 사고 원칙 등
"""
# -------------------------------------------------------------------


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str


def get_openai_client() -> OpenAI:
    """환경변수에서 OPENAI_API_KEY를 읽어와서 클라이언트를 만든다."""
    api_key = os.getenv("OPENAI_API_KEY")

    # 키가 아예 없으면 여기서 바로 에러를 던져준다.
    if not api_key:
        # 여기서 실제 키 값을 찍지는 말자 (보안)
        raise HTTPException(
            status_code=500,
            detail="서버 설정 오류: OPENAI_API_KEY 환경변수가 설정되어 있지 않습니다.",
        )

    return OpenAI(api_key=api_key)


@app.post("/ai-college", response_model=ChatResponse)
def ai_college_chat(req: ChatRequest):
    """커스텀 GPT가 호출할 AI 컬리지 에이전트 API"""
    client = get_openai_client()
    user_message = req.message

    response = client.responses.create(
        model="gpt-4.1-mini",  # 필요하면 gpt-4.1로 변경 가능
        input=[
            {"role": "system", "content": AI_COLLEGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )

    # responses API 결과에서 텍스트만 추출
    reply_text = response.output[0].content[0].text

    return ChatResponse(reply=reply_text)


# (선택) 환경변수 체크용 헬스 체크 엔드포인트
@app.get("/health")
def health():
    has_key = bool(os.getenv("OPENAI_API_KEY"))
    return {"ok": True, "has_openai_key": has_key}
