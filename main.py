from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI

# OpenAI 클라이언트 (OPENAI_API_KEY는 나중에 Railway에서 환경변수로 넣을 것)
client = OpenAI()

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


@app.post("/ai-college", response_model=ChatResponse)
def ai_college_chat(req: ChatRequest):
    """커스텀 GPT가 호출할 AI 컬리지 에이전트 API"""
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
