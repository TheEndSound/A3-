#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Web版个性化学习多智能体系统
FastAPI + SSE + 完整多智能体工作流 + 精美UI
"""

import os
import json
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from sse_starlette.sse import EventSourceResponse
from dotenv import load_dotenv
from graph_web import process_message

load_dotenv()

app = FastAPI(title="AI智学 - 个性化学习多智能体系统")

# 从 index.html 加载前端页面
_INDEX_HTML = None


def _load_index_html() -> str:
    global _INDEX_HTML
    if _INDEX_HTML is not None:
        return _INDEX_HTML
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            _INDEX_HTML = f.read()
        print(f"[前端页面] 已加载: index.html ({len(_INDEX_HTML)} 字符)")
    else:
        _INDEX_HTML = "<html><body><h1>前端页面丢失</h1></body></html>"
    return _INDEX_HTML


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(content=_load_index_html())


@app.post("/chat/stream")
async def chat_stream(request: Request):
    data = await request.json()
    user_message = data.get("message", "")
    session_id = data.get("session_id", "default")

    # 输入验证
    if not user_message or not user_message.strip():
        async def empty_response():
            yield {"data": json.dumps({"type": "status", "content": "⚠️ 消息不能为空，请输入学习相关问题。"})}
            yield {"data": json.dumps({"type": "end"})}
        return EventSourceResponse(empty_response())
    if len(user_message) > 2000:
        async def too_long_response():
            yield {"data": json.dumps({"type": "status", "content": "⚠️ 消息过长（超过2000字符），请精简后重试。"})}
            yield {"data": json.dumps({"type": "end"})}
        return EventSourceResponse(too_long_response())
    if len(session_id) > 128:
        async def bad_session_response():
            yield {"data": json.dumps({"type": "status", "content": "⚠️ session_id 格式无效。"})}
            yield {"data": json.dumps({"type": "end"})}
        return EventSourceResponse(bad_session_response())

    async def event_generator():
        ended = False
        try:
            for event_type, content in process_message(session_id, user_message):
                if await request.is_disconnected():
                    break
                yield {"data": json.dumps({"type": event_type, "content": content})}
                if event_type == "end":
                    ended = True
        except asyncio.CancelledError:
            pass
        except Exception as e:
            yield {"data": json.dumps({"type": "error", "content": str(e)})}
        finally:
            if not ended:
                yield {"data": json.dumps({"type": "end"})}

    return EventSourceResponse(event_generator())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
