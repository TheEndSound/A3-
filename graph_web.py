#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Web版本的多智能体工作流封装（生成器流式版）。
提供 process_message 生成器，实时 yield (event_type, data) 供 SSE 推送。

关键设计：所有 LLM 调用都使用生成器实时产出 token，不等待完整响应。
"""
import os, json, re, requests
from typing import TypedDict, Annotated, Dict, Any, List, Optional
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

load_dotenv()
API_KEY = os.getenv("XF_API_KEY")
URL = "https://spark-api-open.xf-yun.com/v2/chat/completions"

# ================== 课程知识库 ==================
_COURSE_KB = None
def load_course_knowledge_base() -> dict:
    global _COURSE_KB
    if _COURSE_KB is not None: return _COURSE_KB
    kb_path = os.path.join(os.path.dirname(__file__), "course_knowledge_base.json")
    if os.path.exists(kb_path):
        with open(kb_path, "r", encoding="utf-8") as f:
            _COURSE_KB = json.load(f)
        print(f"[课程知识库] 已加载: {_COURSE_KB['course']['name']} ({len(_COURSE_KB['chapters'])} 章)")
    else:
        _COURSE_KB = {"course": {"name": "默认课程"}, "chapters": []}
    return _COURSE_KB

def find_matching_knowledge(query: str, kb: dict, top_k: int = 3) -> List[dict]:
    query_lower = query.lower(); matches = []
    for ch in kb.get("chapters", []):
        ch_score = 5 if (ch["title"].lower() in query_lower or any(word in query_lower for word in ch["title"].split())) else 0
        for kp in ch.get("knowledge_points", []):
            score = ch_score
            kp_name = kp["name"].lower()
            if kp_name in query_lower: score += 3
            elif any(word in query_lower for word in kp_name.split()): score += 1
            if score > 0:
                matches.append({"chapter": ch["title"], "knowledge_point": kp["name"], "difficulty": kp.get("difficulty", "中级"), "score": score})
    matches.sort(key=lambda x: x["score"], reverse=True)
    seen = set(); unique = []
    for m in matches:
        if m["knowledge_point"] not in seen: seen.add(m["knowledge_point"]); unique.append(m)
    return unique[:top_k]

def get_course_context(query: str) -> str:
    kb = load_course_knowledge_base(); matches = find_matching_knowledge(query, kb)
    if not matches: return ""
    ctx = f"【课程上下文】来自《{kb['course']['name']}》\n"; matched = set()
    for m in matches:
        if m["chapter"] not in matched:
            matched.add(m["chapter"])
            for ch in kb["chapters"]:
                if ch["title"] == m["chapter"]:
                    ctx += f"\n📖 {ch['title']}（{ch.get('difficulty','中级')}）：{'; '.join(ch.get('learning_objectives',[]))}\n"; break
    ctx += "\n匹配知识点：\n"
    for m in matches[:5]: ctx += f"  - {m['knowledge_point']}（{m['difficulty']}）\n"
    return ctx

# ================== 工具函数 ==================
def extract_json(text: str) -> Optional[dict]:
    if not text: return None
    try: return json.loads(text.strip())
    except Exception: pass
    depth = 0; start = -1
    for i, ch in enumerate(text):
        if ch == '{':
            if depth == 0: start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start != -1:
                try: return json.loads(text[start:i+1])
                except Exception: pass
    for m in re.finditer(r'\{[^{}]*\}', text, re.DOTALL):
        try: return json.loads(m.group())
        except Exception:
            try: return json.loads(re.sub(r',\s*([}\]])', r'\1', m.group()))
            except Exception: continue
    return None

SENSITIVE_KEYWORDS = ['政治敏感', '色情', '暴力', '违法', '毒品', '枪支', '赌博']
_SENSITIVE_PATTERNS = [re.compile(re.escape(kw), re.IGNORECASE) for kw in SENSITIVE_KEYWORDS]
def content_safety_check(text: str) -> Optional[str]:
    for p, kw in zip(_SENSITIVE_PATTERNS, SENSITIVE_KEYWORDS):
        if p.search(text): return f"抱歉，您输入的内容包含敏感词汇（{kw}），请重新提问。"
    return None
def output_safety_filter(text: str) -> str:
    for p, kw in zip(_SENSITIVE_PATTERNS, SENSITIVE_KEYWORDS):
        if p.search(text): return "⚠️ 模型生成的内容包含不安全信息，已过滤。请重新尝试。"
    return text

# ================== 生成器版流式LLM调用 ==================
def call_llm_gen(messages):
    """
    生成器：实时产出来自讯飞星火的 token。
    每次 yield ("chunk", content_text).
    迭代结束后，通过 .full_response 获取完整回复。
    """
    headers = {'Authorization': API_KEY, 'content-type': 'application/json'}
    body = {
        "model": "x1", "user": "web_user", "messages": messages, "stream": True,
        "tools": [{"type": "web_search", "web_search": {"enable": True, "search_mode": "deep"}}]
    }
    full = ""
    reasoning_buf = ""
    thinking_shown = False
    reasoning_yielded = False
    try:
        resp = requests.post(URL, json=body, headers=headers, stream=True, timeout=120)
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line or line == b'[DONE]': continue
            s = line.decode('utf-8')
            if s.startswith('data:'): s = s[5:]
            try:
                d = json.loads(s)
                delta = d['choices'][0]['delta']
                if 'reasoning_content' in delta and delta['reasoning_content']:
                    reasoning_buf += delta['reasoning_content']
                    if not thinking_shown and len(reasoning_buf) > 10:
                        yield ("status", "🧠 模型思考中...")
                        thinking_shown = True
                if 'content' in delta and delta.get('content'):
                    if reasoning_buf and not reasoning_yielded:
                        yield ("status", f"🧠 思考完成，正在生成回复...")
                        reasoning_yielded = True
                    full += delta['content']
                    yield ("chunk", delta['content'])
            except Exception: pass
    except Exception as e:
        yield ("error", str(e))
    yield ("_llm_done", full)  # 用 _llm_done 避免与 run_graph 的 _done 冲突  # 特殊事件传递完整回复

def call_llm_sync(messages):
    """非流式版，返回完整文本。"""
    headers = {'Authorization': API_KEY, 'content-type': 'application/json'}
    body = {
        "model": "x1", "user": "web_user", "messages": messages, "stream": True,
        "tools": [{"type": "web_search", "web_search": {"enable": True, "search_mode": "deep"}}]
    }
    full = ""
    try:
        resp = requests.post(URL, json=body, headers=headers, stream=True, timeout=120)
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line or line == b'[DONE]': continue
            s = line.decode('utf-8')
            if s.startswith('data:'): s = s[5:]
            try:
                d = json.loads(s)
                if 'content' in d['choices'][0].get('delta', {}):
                    full += d['choices'][0]['delta']['content']
            except Exception: pass
    except Exception: pass
    return full

# ================== 状态定义 ==================
class StudentProfile(TypedDict):
    knowledge_base: str; learning_style: str; weak_points: List[str]
    interest: str; learning_pace: str; interaction_summary: str

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    user_intent: str; profile: Optional[StudentProfile]
    learning_plan: Optional[List[Dict]]
    course_context: Optional[str]; resource_plan: Optional[str]

# ================== 智能体节点（生成器版） ==================
def is_tutor_request(text: str) -> bool:
    return any(kw in text for kw in ["为什么", "我不懂", "帮我解释", "什么意思", "怎么理解", "讲解一下", "辅导我", "帮我分析"])

def profile_agent(state, events_out):
    """画像分析智能体（非流式，一次性输出）。"""
    existing = state.get("profile") or {}
    recent = state["messages"][-4:]
    ctx = "\n".join([f"{'用户' if isinstance(m, HumanMessage) else 'AI'}: {m.content}" for m in recent])
    hit = content_safety_check(ctx)
    if hit:
        events_out.append(("status", hit))
        return {"profile": existing}
    prompt = f"""你是一个教育画像分析专家。根据以下对话历史，提取学生的画像信息（JSON格式）。
已有画像：{json.dumps(existing, ensure_ascii=False)}
对话历史：{ctx}
输出必须是合法JSON，包含6个字段：
- "knowledge_base": "初级"/"中级"/"高级"
- "learning_style": "视觉型"/"听觉型"/"动手型"
- "weak_points": 字符串数组
- "interest": 字符串
- "learning_pace": "快"/"中"/"慢"
- "interaction_summary": 简短一句话总结
只输出JSON对象，不要有任何其他解释或标记。"""
    resp = call_llm_sync([{"role": "user", "content": prompt}])
    new_p = extract_json(resp)
    if new_p is not None:
        merged = {**existing, **new_p}
        if "weak_points" in new_p and isinstance(new_p["weak_points"], list):
            existing_weak = existing.get("weak_points", [])
            if isinstance(existing_weak, list):
                merged["weak_points"] = list(set(existing_weak + new_p["weak_points"]))
        events_out.append(("status", f"✅ 画像已更新：{merged.get('knowledge_base','?')}基础 | {merged.get('learning_style','?')}型"))
        events_out.append(("data", json.dumps({"type": "profile", "data": merged})))
        return {"profile": merged}
    events_out.append(("status", "ℹ️ 画像分析完成（基于已有信息）"))
    return {"profile": existing}

def classify_intent(state, events_out):
    last = state["messages"][-1].content
    prompt = f"""判断用户输入目的，只输出一个词语（greeting / resource / chat / tutor / evaluation）：
- greeting: 纯打招呼问好（如你好、嗨、早上好），不包括询问身份
- resource: 请求生成学习资源
- tutor: 请求辅导或解释，包含"为什么"、"我不懂"、"帮我解释"等疑问
- evaluation: 请求学习评估，包含"评估"、"测试我"、"考核"、"我的学习效果"等关键词
- chat: 普通学术问题、闲聊、询问身份（如你是谁、你叫什么）、或任何带具体问题意图的输入
输入: {last}
输出:"""
    resp = call_llm_sync([{"role": "user", "content": prompt}]).strip().lower()
    if "greeting" in resp: intent = "greeting"
    elif "resource" in resp: intent = "resource"
    elif "tutor" in resp: intent = "tutor"
    elif "evaluation" in resp: intent = "evaluation"
    else: intent = "chat"
    names = {"greeting":"问候","resource":"资源生成","chat":"普通对话","tutor":"智能辅导","evaluation":"效果评估"}
    events_out.append(("status", f"🔍 意图识别：{names.get(intent,intent)}"))
    return {"user_intent": intent}

# ----- 流式节点（生成器） -----
def _build_identity_prompt(user_msg: str) -> str:
    """构建带身份说明的prompt，确保模型知道自己是AI智学。"""
    return f"""你的身份是「AI智学」，一个智能学习助手，由多智能体系统驱动，提供个性化学习服务。
你的能力包括：生成学习资源（文档、导图、练习、案例、视频）、智能辅导、学习路径规划和效果评估。

用户说：{user_msg}

请根据用户输入自然回应：
- 如果用户问好（你好/嗨/早上好），热情问候；
- 如果用户问你是谁/你叫什么，请自我介绍（名称、身份、能做什么）；
- 否则正常回答问题。
回复简短自然（不超过80字）。"""

def greeting_node_gen(state):
    """问候节点生成器（流式输出）。"""
    yield ("status", "👋 正在准备问候...")
    user_msg = state["messages"][-1].content if state["messages"] else ""
    prompt = _build_identity_prompt(user_msg)
    answer = ""
    for evt in call_llm_gen([{"role": "user", "content": prompt}]):
        if evt[0] == "chunk": answer += evt[1]
        yield evt
    answer = output_safety_filter(answer)
    yield ("_result", {"messages": [AIMessage(content=answer)]})

def chat_node_gen(state):
    """普通对话节点生成器（实时流式输出）。"""
    profile = state.get("profile") or {}
    course_ctx = state.get("course_context", "")
    history = []
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage): history.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage): history.append({"role": "assistant", "content": msg.content})

    # 检查用户是否在询问身份
    last_msg = history[-1]["content"] if history else ""
    identity_keywords = ["你是谁", "你叫什么", "你是什么", "你的名字", "你是哪个", "你的身份"]
    is_asking_identity = any(kw in last_msg for kw in identity_keywords)

    style_guide = {"视觉型":"多使用图表和结构化格式","听觉型":"多用类比和叙述","动手型":"提供可运行代码示例"}.get(profile.get("learning_style","视觉型"),"")
    sysp = f"""你是一个智能学习助手「AI智学」，提供个性化学习服务。
根据学生画像调整回答：
- 知识基础：{profile.get('knowledge_base','初级')}
- 学习风格：{profile.get('learning_style','视觉型')}，{style_guide}
- 薄弱点：{', '.join(profile.get('weak_points',['无']))}
{course_ctx}"""

    # 如果用户在问身份，把身份声明直接注入到用户消息之前
    if is_asking_identity:
        identity_note = {"role": "user", "content": "【注意】你的名字是「AI智学」，是一个智能学习助手。如果用户问你是谁，请直接告诉用户你的名字和功能。"}
        msgs = [{"role":"system","content": sysp}, identity_note] + history
    else:
        msgs = [{"role":"system","content": sysp}] + history

    answer = ""
    for evt in call_llm_gen(msgs):
        if evt[0] == "chunk": answer += evt[1]
        yield evt  # 实时转发 chunk
    answer = output_safety_filter(answer)
    plan = state.get("learning_plan")
    if plan:
        pt = "\n\n📌 **推荐学习路径：**\n" + "\n".join(f"{i+1}. {s['title']}" for i,s in enumerate(plan))
        answer += pt
        yield ("chunk", pt)
    yield ("_result", {"messages": [AIMessage(content=answer)]})

def tutor_node_gen(state):
    """智能辅导节点生成器。"""
    last_q = state["messages"][-1].content
    profile = state.get("profile") or {}
    course_ctx = state.get("course_context", "")
    yield ("status", "🧑‍🏫 智能辅导节点启动，正在为你详细解答...")
    prompt = f"""你是一个有耐心的智能辅导老师。请提供多模态辅导：
1. **文字解答**：通俗易懂的解释
2. **图解说明**：ASCII图、流程图或类比
3. **短视频脚本建议**：30-60秒内容脚本
4. **实践建议**：可动手尝试的练习
学生画像：知识基础={profile.get('knowledge_base','初级')} | 风格={profile.get('learning_style','视觉型')} | 薄弱点={', '.join(profile.get('weak_points',['无']))}
{course_ctx}
学生提问：{last_q}
请在回复前标注 **[智能辅导]**。"""
    answer = ""
    for evt in call_llm_gen([{"role":"user","content": prompt}]):
        if evt[0] == "chunk": answer += evt[1]
        yield evt
    answer = output_safety_filter(answer)
    yield ("_result", {"messages": [AIMessage(content=answer)]})

def evaluation_node_gen(state):
    """效果评估节点生成器。"""
    profile = state.get("profile") or {}
    yield ("status", "📊 学习效果评估节点启动，正在分析学习情况...")
    hist = ""
    for msg in state["messages"][-10:]:
        role = "用户" if isinstance(msg, HumanMessage) else "AI"
        hist += f"{role}: {msg.content[:200]}\n"
    prompt = f"""你是一个学习效果评估专家。根据对话历史和画像生成评估报告。
学生画像：知识基础={profile.get('knowledge_base','初级')} | 薄弱点={', '.join(profile.get('weak_points',['无']))}
最近对话：{hist}
包含：1.知识掌握度 2.薄弱环节 3.进步情况 4.学习建议 5.调整策略
在回复前标注 **[学习效果评估]**。"""
    answer = ""
    for evt in call_llm_gen([{"role":"user","content": prompt}]):
        if evt[0] == "chunk": answer += evt[1]
        yield evt
    answer = output_safety_filter(answer)
    yield ("_result", {"messages": [AIMessage(content=answer)]})

def resource_planner_agent(state, events_out):
    """资源规划智能体（非流式）。"""
    profile = state.get("profile") or {}
    last_q = state["messages"][-1].content
    course_ctx = state.get("course_context", "")
    events_out.append(("status", "📋 资源规划智能体：正在分析需求..."))
    prompt = f"""你是一个学习资源规划专家。根据学生画像和课程上下文，规划资源内容方向。
学生画像：知识基础={profile.get('knowledge_base','初级')} | 风格={profile.get('learning_style','视觉型')}
薄弱点={', '.join(profile.get('weak_points',['无']))} | 兴趣={profile.get('interest','编程')}
{course_ctx}
学生提问：{last_q}
输出JSON：{{"topic":"核心主题","teaching_approach":"教学方法","focus_areas":["重点1","重点2"],"difficulty_level":"难度","style_recommendation":"呈现建议"}}
只输出JSON对象。"""
    resp = call_llm_sync([{"role":"user","content": prompt}])
    plan = extract_json(resp)
    if plan:
        events_out.append(("status", f"📋 规划完成：主题「{plan.get('topic','未知')}」"))
        return {"resource_plan": json.dumps(plan, ensure_ascii=False), "_resource_msg_index": len(state["messages"])}
    events_out.append(("status", "📋 资源规划完成"))
    return {"resource_plan": resp, "_resource_msg_index": len(state["messages"])}

def content_generator_gen(state):
    """内容生成智能体生成器：生成讲解文档+练习题（流式）。"""
    profile = state.get("profile") or {}
    last_q = state["messages"][-1].content
    course_ctx = state.get("course_context", "")
    rp = state.get("resource_plan", "")
    yield ("status", "📝 内容生成智能体：正在生成讲解文档和练习题...")
    prompt = f"""你是一个教学内容生成专家。生成讲解文档和练习题。
资源规划：{rp}
{course_ctx}
学生画像：知识基础={profile.get('knowledge_base','初级')} | 风格={profile.get('learning_style','视觉型')}
薄弱点={', '.join(profile.get('weak_points',['无']))}
学生提问：{last_q}
格式：
## 📘 1. 讲解文档（概念解释+原理+代码示例+复杂度分析+应用场景）
## 📝 2. 练习题（至少3道：选择题+编程题，附答案和解析）"""
    answer = ""
    for evt in call_llm_gen([{"role":"user","content": prompt}]):
        if evt[0] == "chunk": answer += evt[1]
        yield evt
    answer = output_safety_filter(answer)
    yield ("_result", {"messages": [AIMessage(content=answer)]})

def multimodal_generator_gen(state):
    """多模态设计智能体生成器：思维导图+视频脚本（流式）。"""
    profile = state.get("profile") or {}
    last_q = state["messages"][-1].content
    course_ctx = state.get("course_context", "")
    rp = state.get("resource_plan", "")
    yield ("status", "🎨 多模态设计智能体：正在生成思维导图和视频脚本...")
    prompt = f"""你是一个多模态教学设计专家。生成思维导图和视频脚本。
资源规划：{rp}
{course_ctx}
学生画像：知识基础={profile.get('knowledge_base','初级')} | 风格={profile.get('learning_style','视觉型')}
薄弱点={', '.join(profile.get('weak_points',['无']))}
学生提问：{last_q}
格式：
## 🗺️ 3. 知识点思维导图（文本描述，至少3个一级分支）
## 🎥 5. 多模态教学视频/动画脚本（3-5个场景，含画面描述+旁白+动画效果+时长）"""
    answer = ""
    for evt in call_llm_gen([{"role":"user","content": prompt}]):
        if evt[0] == "chunk": answer += evt[1]
        yield evt
    answer = output_safety_filter(answer)
    yield ("_result", {"messages": [AIMessage(content=answer)]})

def case_generator_gen(state):
    """案例生成智能体生成器：实操案例（流式）。"""
    profile = state.get("profile") or {}
    last_q = state["messages"][-1].content
    course_ctx = state.get("course_context", "")
    rp = state.get("resource_plan", "")
    yield ("status", "💻 案例生成智能体：正在生成实操案例...")
    prompt = f"""你是一个实践教学案例设计专家。生成实操案例。
资源规划：{rp}
{course_ctx}
学生画像：知识基础={profile.get('knowledge_base','初级')} | 风格={profile.get('learning_style','视觉型')}
薄弱点={', '.join(profile.get('weak_points',['无']))} | 兴趣={profile.get('interest','编程')}
学生提问：{last_q}
格式：
## 💻 4. 实操案例（案例名称+问题描述+需求分析+完整代码+运行结果+扩展思考）
代码必须完整可运行。"""
    answer = ""
    for evt in call_llm_gen([{"role":"user","content": prompt}]):
        if evt[0] == "chunk": answer += evt[1]
        yield evt
    answer = output_safety_filter(answer)
    yield ("_result", {"messages": [AIMessage(content=answer)]})

def resource_merger_agent(state, events_out):
    """资源合并智能体（非流式）。"""
    parts = []
    msg_start = state.get("_resource_msg_index", 0)
    for msg in reversed(state["messages"][msg_start:]):
        if isinstance(msg, AIMessage):
            if any(h in msg.content for h in ["讲解文档","知识点思维导图","练习题","实操案例","多模态教学视频"]):
                parts.append(msg.content)
                if len(parts) >= 4: break
    combined = "\n\n".join(reversed(parts))
    events_out.append(("status", "✅ 资源合并完成！"))
    headers = ["讲解文档","知识点思维导图","练习题","实操案例","多模态教学视频"]
    missing = [h for h in headers if h not in combined]
    if missing:
        events_out.append(("status", f"⚠️ 补充生成: {missing}"))
        sup = call_llm_sync([{"role":"user","content": f"请补充以下资源：{missing}"}])
        combined += "\n\n--- 补充 ---\n" + sup
    return {"messages": [AIMessage(content=combined)]}

def plan_node(state, events_out):
    """路径规划智能体（非流式）。"""
    profile = state.get("profile") or {}
    course_ctx = state.get("course_context", "")
    weak = ', '.join(profile.get('weak_points',[])) or '无'
    events_out.append(("status", "🗺️ 路径规划智能体：正在生成个性化学习路径..."))
    prompt = f"""根据学生画像和课程知识，规划个性化学习路径（3-5个步骤）。
画像：知识基础={profile.get('knowledge_base','初级')} | 风格={profile.get('learning_style','视觉型')}
薄弱点={weak} | 兴趣={profile.get('interest','编程')} | 节奏={profile.get('learning_pace','中')}
{course_ctx}
要求：针对薄弱点，每个步骤包含具体活动和目标，推荐资源类型。
输出JSON对象，包含两个字段：
1. "summary": 路径总结（一句话）
2. "steps": 步骤数组，每个元素包含：
   - "title": 步骤标题（简短）
   - "description": 详细描述（包含具体活动）
   - "goal": 学习目标
   - "resource_types": 推荐资源类型数组，如 ["文档", "视频", "练习", "案例"]

示例输出：
{{"summary": "从基础到实践的系统学习路径", "steps": [
  {{"title": "夯实基础", "description": "复习核心概念和基本原理", "goal": "建立知识框架", "resource_types": ["文档", "视频"]}},
  {{"title": "专项练习", "description": "针对薄弱点进行针对性练习", "goal": "攻克薄弱环节", "resource_types": ["练习", "案例"]}}
]}}

只输出JSON对象，不要有任何其他解释或标记。"""
    plan_text = call_llm_sync([{"role":"user","content": prompt}])
    parsed = extract_json(plan_text)
    if parsed and isinstance(parsed, dict) and "steps" in parsed:
        steps = parsed["steps"]
        summary = parsed.get("summary", "个性化学习路径")
    else:
        # 回退到文本解析
        steps = []
        summary = "个性化学习路径"
        for line in plan_text.split('\n'):
            line = line.strip()
            if line and len(line)>2 and line[0].isdigit() and '.' in line:
                s = line.split('.',1)[1].strip()
                if s:
                    steps.append({"title": s[:20], "description": s, "goal": "掌握相关知识", "resource_types": ["文档", "练习"]})
    if not steps:
        steps = [
            {"title": "夯实基础", "description": f"复习核心概念和基本原理，重点关注{weak}", "goal": "建立知识框架", "resource_types": ["文档", "视频"]},
            {"title": "专项练习", "description": f"针对薄弱点进行针对性练习和巩固", "goal": "攻克薄弱环节", "resource_types": ["练习", "案例"]},
            {"title": "综合实践", "description": "完成综合案例，将知识融会贯通", "goal": "综合运用", "resource_types": ["案例", "练习"]}
        ]
    # 设置状态：第一步为current，其余为todo
    for i, step in enumerate(steps):
        step["status"] = "current" if i == 0 else "todo"
    events_out.append(("status", f"✅ 已生成 {len(steps)} 个学习步骤"))
    events_out.append(("data", json.dumps({"type": "plan", "data": {"steps": steps, "summary": summary}})))
    return {"learning_plan": steps}

# ================== 主流程生成器 ==================
def run_graph(state):
    """
    运行整个多智能体工作流，实时 yield 事件。
    每个 yield 格式: ("chunk", text) / ("status", text) / ("_result", dict) / ("_done", "")
    普通事件直接转发给前端；_result 用于内部状态更新；_done 表示工作流结束。
    """
    # 阶段1：画像分析（非流式）
    events_buf = []
    pr = profile_agent(state, events_buf)
    for e in events_buf: yield e
    state.update(pr)

    # 阶段2：意图识别（非流式）
    events_buf.clear()
    ir = classify_intent(state, events_buf)
    for e in events_buf: yield e
    state.update(ir)

    intent = state["user_intent"]

    # 阶段3：按意图路由
    if intent == "greeting":
        for evt in greeting_node_gen(state):
            if evt[0] == "_result": state.update(evt[1])
            elif evt[0] == "_llm_done": pass
            else: yield evt

    elif intent == "resource":
        # 资源规划（非流式）
        events_buf.clear()
        rp = resource_planner_agent(state, events_buf)
        for e in events_buf: yield e
        state.update(rp)

        # 内容生成（流式）
        for evt in content_generator_gen(state):
            if evt[0] == "_result":
                state["messages"].append(evt[1]["messages"][0])
            elif evt[0] == "_llm_done": pass
            else: yield evt

        # 多模态设计（流式）
        for evt in multimodal_generator_gen(state):
            if evt[0] == "_result":
                state["messages"].append(evt[1]["messages"][0])
            elif evt[0] == "_llm_done": pass
            else: yield evt

        # 案例生成（流式）
        for evt in case_generator_gen(state):
            if evt[0] == "_result":
                state["messages"].append(evt[1]["messages"][0])
            elif evt[0] == "_llm_done": pass
            else: yield evt

        # 合并（非流式）
        events_buf.clear()
        mr = resource_merger_agent(state, events_buf)
        for e in events_buf: yield e
        if "messages" in mr and mr["messages"]:
            state["messages"].append(mr["messages"][0])

        # 路径规划（非流式）
        events_buf.clear()
        pl = plan_node(state, events_buf)
        for e in events_buf: yield e
        state.update(pl)

    elif intent == "tutor":
        for evt in tutor_node_gen(state):
            if evt[0] == "_result": state.update(evt[1])
            elif evt[0] == "_llm_done": pass
            else: yield evt

    elif intent == "evaluation":
        for evt in evaluation_node_gen(state):
            if evt[0] == "_result": state.update(evt[1])
            elif evt[0] == "_llm_done": pass
            else: yield evt

    else:  # chat
        for evt in chat_node_gen(state):
            if evt[0] == "_result": state.update(evt[1])
            elif evt[0] == "_llm_done": pass
            else: yield evt

    yield ("_done", "")


# ================== 对外接口 ==================
sessions: Dict[str, AgentState] = {}
_session_times: Dict[str, float] = {}  # LRU tracking

def process_message(session_id: str, user_message: str):
    """处理用户消息，实时 yield 流式事件。"""
    import time
    if session_id not in sessions:
        sessions[session_id] = {"messages":[],"user_intent":"","profile":None,"learning_plan":None,"course_context":"","resource_plan":""}
    _session_times[session_id] = time.time()
    # LRU 风格：优先驱逐最不活跃的 session
    if len(sessions) > 50:
        oldest = min(_session_times, key=lambda k: _session_times[k])
        if oldest in sessions:
            del sessions[oldest]
            del _session_times[oldest]
    state = sessions[session_id]

    hit = content_safety_check(user_message)
    if hit:
        yield ("status", hit); yield ("end", ""); return

    course_ctx = get_course_context(user_message)
    if course_ctx: state["course_context"] = course_ctx

    # 消息历史裁剪：超过 50 条时保留最近的 30 条
    human_count = sum(1 for m in state["messages"] if isinstance(m, HumanMessage))
    if human_count >= 50:
        state["messages"] = state["messages"][-30:]

    state["messages"].append(HumanMessage(content=user_message))

    try:
        for evt in run_graph(state):
            if evt[0] == "_done":
                break
            yield evt  # 实时转发 chunk/status 到前端
        yield ("end", "")
    except Exception as e:
        # 异常发生时回滚已添加的 HumanMessage
        if state["messages"] and isinstance(state["messages"][-1], HumanMessage) and state["messages"][-1].content == user_message:
            state["messages"].pop()
        yield ("status", f"❌ 处理出错：{str(e)}")
        yield ("error", str(e))
