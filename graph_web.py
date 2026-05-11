#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Web版多智能体工作流封装（生成器流式版）。
提供 process_message 生成器，实时 yield (event_type, data) 供 SSE 推送。

设计：所有 LLM 调用使用生成器实时产出 token，通过 SSE 推送到前端。
"""

import json
import time
import threading
from typing import Dict, Any, List, Optional

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from core import (
    get_course_context,
    extract_json,
    content_safety_check,
    output_safety_filter,
    call_llm_stream,
    call_llm_sync,
    search_bilibili_videos,
    build_profile_prompt,
    build_intent_prompt,
    build_greeting_prompt,
    build_chat_system_prompt,
    build_tutor_prompt,
    build_evaluation_prompt,
    build_plan_prompt,
    is_tutor_request,
    is_identity_question,
    merge_profile,
    AgentState,
    RESOURCE_HEADERS,
    logger,
)


# ================== 智能体节点（生成器版） ==================

def profile_agent(state: AgentState, events_out: List):
    """画像分析智能体（非流式）。"""
    existing = state.get("profile") or {}
    recent = state["messages"][-4:]
    ctx = "\n".join(
        [f"{'用户' if isinstance(m, HumanMessage) else 'AI'}: {m.content}" for m in recent]
    )

    hit = content_safety_check(ctx)
    if hit:
        events_out.append(("status", hit))
        return {"profile": existing}

    prompt = build_profile_prompt(existing, ctx)
    resp = call_llm_sync([{"role": "user", "content": prompt}])
    new_p = extract_json(resp)

    if new_p is not None:
        merged = merge_profile(existing, new_p)
        events_out.append(("status", f"✅ 画像已更新：{merged.get('knowledge_base', '?')}基础 | {merged.get('learning_style', '?')}型"))
        events_out.append(("data", json.dumps({"type": "profile", "data": merged})))
        return {"profile": merged}

    events_out.append(("status", "ℹ️ 画像分析完成（基于已有信息）"))
    return {"profile": existing}


def classify_intent(state: AgentState, events_out: List):
    """意图识别智能体（非流式）。"""
    last = state["messages"][-1].content
    prompt = build_intent_prompt(last)
    resp = call_llm_sync([{"role": "user", "content": prompt}]).strip().lower()

    if "greeting" in resp:
        intent = "greeting"
    elif "resource" in resp:
        intent = "resource"
    elif "tutor" in resp:
        intent = "tutor"
    elif "evaluation" in resp:
        intent = "evaluation"
    else:
        intent = "chat"

    names = {"greeting": "问候", "resource": "资源生成", "chat": "普通对话", "tutor": "智能辅导", "evaluation": "效果评估"}
    events_out.append(("status", f"🔍 意图识别：{names.get(intent, intent)}"))
    return {"user_intent": intent}


# ----- 流式节点（生成器） -----

def greeting_node_gen(state: AgentState):
    """问候节点生成器（流式输出）。"""
    yield ("status", "👋 正在准备问候...")
    user_msg = state["messages"][-1].content if state["messages"] else ""
    prompt = build_greeting_prompt(user_msg)

    answer = ""
    for evt in call_llm_stream([{"role": "user", "content": prompt}]):
        if evt[0] == "chunk":
            answer += evt[1]
        yield evt
    answer = output_safety_filter(answer)
    yield ("_result", {"messages": [AIMessage(content=answer)]})


def chat_node_gen(state: AgentState):
    """普通对话节点生成器（实时流式输出）。"""
    profile = state.get("profile") or {}
    course_ctx = state.get("course_context", "")

    history = []
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage):
            history.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            history.append({"role": "assistant", "content": msg.content})

    sysp = build_chat_system_prompt(profile, course_ctx)

    last_msg = history[-1]["content"] if history else ""
    if is_identity_question(last_msg):
        identity_note = {
            "role": "user",
            "content": "【注意】你的名字是「AI智学」，是一个智能学习助手。如果用户问你是谁，请直接告诉用户你的名字和功能。",
        }
        msgs = [{"role": "system", "content": sysp}, identity_note] + history
    else:
        msgs = [{"role": "system", "content": sysp}] + history

    answer = ""
    for evt in call_llm_stream(msgs):
        if evt[0] == "chunk":
            answer += evt[1]
        yield evt

    answer = output_safety_filter(answer)

    # 追加学习路径推荐
    plan = state.get("learning_plan")
    if plan:
        pt = "\n\n📌 **推荐学习路径：**\n" + "\n".join(
            f"{i + 1}. {s.get('title', str(s))}" for i, s in enumerate(plan)
        )
        answer += pt
        yield ("chunk", pt)

    yield ("_result", {"messages": [AIMessage(content=answer)]})


def tutor_node_gen(state: AgentState):
    """智能辅导节点生成器（流式输出）。"""
    last_q = state["messages"][-1].content
    profile = state.get("profile") or {}
    course_ctx = state.get("course_context", "")

    yield ("status", "🧑‍🏫 智能辅导节点启动，正在为你详细解答...")
    prompt = build_tutor_prompt(last_q, profile, course_ctx)

    answer = ""
    for evt in call_llm_stream([{"role": "user", "content": prompt}]):
        if evt[0] == "chunk":
            answer += evt[1]
        yield evt
    answer = output_safety_filter(answer)
    yield ("_result", {"messages": [AIMessage(content=answer)]})


def evaluation_node_gen(state: AgentState):
    """效果评估节点生成器（流式输出）。"""
    profile = state.get("profile") or {}

    yield ("status", "📊 学习效果评估节点启动，正在分析学习情况...")
    hist = ""
    for msg in state["messages"][-10:]:
        role = "用户" if isinstance(msg, HumanMessage) else "AI"
        hist += f"{role}: {msg.content[:200]}\n"

    prompt = build_evaluation_prompt(profile, hist)

    answer = ""
    for evt in call_llm_stream([{"role": "user", "content": prompt}]):
        if evt[0] == "chunk":
            answer += evt[1]
        yield evt
    answer = output_safety_filter(answer)

    # 尝试提取结构化评估数据
    eval_data = extract_json(answer)
    if eval_data and "overall_score" in eval_data:
        yield ("data", json.dumps({"type": "eval", "data": eval_data}))

    yield ("_result", {"messages": [AIMessage(content=answer)]})


# ================== 资源生成多智能体协作组 ==================

def resource_planner_agent(state: AgentState, events_out: List):
    """资源规划智能体（非流式）。"""
    profile = state.get("profile") or {}
    last_q = state["messages"][-1].content
    course_ctx = state.get("course_context", "")

    events_out.append(("status", "📋 资源规划智能体：正在分析需求..."))
    prompt = f"""你是一个学习资源规划专家。根据学生画像和课程上下文，规划资源内容方向。
学生画像：知识基础={profile.get('knowledge_base', '初级')} | 风格={profile.get('learning_style', '视觉型')}
薄弱点={', '.join(profile.get('weak_points', ['无']))} | 兴趣={profile.get('interest', '编程')}
{course_ctx}
学生提问：{last_q}
输出JSON：{{"topic":"核心主题","teaching_approach":"教学方法","focus_areas":["重点1","重点2"],"difficulty_level":"难度","style_recommendation":"呈现建议"}}
只输出JSON对象。"""

    resp = call_llm_sync([{"role": "user", "content": prompt}])
    plan = extract_json(resp)
    if plan:
        events_out.append(("status", f"📋 规划完成：主题「{plan.get('topic', '未知')}」"))
        return {"resource_plan": json.dumps(plan, ensure_ascii=False), "_resource_msg_index": len(state["messages"])}
    events_out.append(("status", "📋 资源规划完成"))
    return {"resource_plan": resp, "_resource_msg_index": len(state["messages"])}


def _build_resource_prompt_compact(agent_title: str, task: str, profile, course_ctx: str, rp: str, last_q: str, fmt: str) -> str:
    """构建资源生成类智能体的 prompt（紧凑版）。"""
    return f"""你是一个{agent_title}。{task}
资源规划：{rp}
{course_ctx}
学生画像：知识基础={profile.get('knowledge_base', '初级')} | 风格={profile.get('learning_style', '视觉型')}
薄弱点={', '.join(profile.get('weak_points', ['无']))} | 兴趣={profile.get('interest', '编程')}
学生提问：{last_q}
格式：
{fmt}"""


def content_generator_gen(state: AgentState):
    """内容生成智能体生成器：讲解文档+练习题（流式）。"""
    profile = state.get("profile") or {}
    yield ("status", "📝 内容生成智能体：正在生成讲解文档和练习题...")

    prompt = _build_resource_prompt_compact(
        "教学内容生成专家", "生成讲解文档和练习题。",
        profile,
        state.get("course_context", ""),
        state.get("resource_plan", ""),
        state["messages"][-1].content,
        "## 📘 1. 讲解文档（概念解释+原理+代码示例+复杂度分析+应用场景）\n"
        "## 📝 2. 练习题（至少3道：选择题+编程题，附答案和解析）",
    )

    answer = ""
    for evt in call_llm_stream([{"role": "user", "content": prompt}]):
        if evt[0] == "chunk":
            answer += evt[1]
        yield evt
    answer = output_safety_filter(answer)
    yield ("_result", {"messages": [AIMessage(content=answer)]})


def multimodal_generator_gen(state: AgentState):
    """多模态设计智能体生成器：思维导图+视频脚本（流式）。"""
    profile = state.get("profile") or {}
    yield ("status", "🎨 多模态设计智能体：正在生成思维导图和视频脚本...")

    prompt = _build_resource_prompt_compact(
        "多模态教学设计专家", "生成知识点思维导图（Mermaid格式）和视频脚本。",
        profile,
        state.get("course_context", ""),
        state.get("resource_plan", ""),
        state["messages"][-1].content,
        "## 🗺️ 3. 知识点思维导图\n"
        "输出 Mermaid mindmap 代码块（必须用 ```mermaid 包裹），格式示例：\n"
        "```mermaid\n"
        "mindmap\n"
        "  root((核心主题))\n"
        "    分支1\n"
        "      子节点A\n"
        "      子节点B\n"
        "    分支2\n"
        "      子节点C\n"
        "      子节点D\n"
        "```\n"
        "要求：至少3个一级分支，每分支至少2个子节点，根节点用 ((双括号)) 包裹。\n"
        "## 🎥 5. 多模态教学视频/动画脚本（3-5个场景，含画面描述+旁白+动画效果+时长）",
    )

    answer = ""
    for evt in call_llm_stream([{"role": "user", "content": prompt}]):
        if evt[0] == "chunk":
            answer += evt[1]
        yield evt
    answer = output_safety_filter(answer)

    # 搜索B站相关视频，追加 HTML 注释标记
    try:
        rp_raw = state.get("resource_plan", "")
        plan = extract_json(rp_raw) if rp_raw else {}
        topic = plan.get("topic", "") if isinstance(plan, dict) else ""
        # 回退：尝试从 resource_plan 字符串中正则提取 topic
        if not topic:
            import re
            m = re.search(r'"topic"\s*:\s*"([^"]+)"', rp_raw) if rp_raw else None
            topic = m.group(1) if m else ""
        # 再次回退：使用用户提问中的关键词
        if not topic:
            last_msg = state["messages"][-1].content if state["messages"] else ""
            topic = last_msg[:50]  # 截取前50字符作为搜索词
        if topic:
            videos = search_bilibili_videos(topic)
            if videos:
                answer += "\n\n---\n\n## 🎬 相关视频资源\n"
                for v in videos:
                    answer += f"\n<!-- BILIBILI_VIDEO {json.dumps(v, ensure_ascii=False)} -->"
    except Exception:
        pass

    yield ("_result", {"messages": [AIMessage(content=answer)]})


def case_generator_gen(state: AgentState):
    """案例生成智能体生成器：实操案例（流式）。"""
    profile = state.get("profile") or {}
    yield ("status", "💻 案例生成智能体：正在生成实操案例...")

    prompt = _build_resource_prompt_compact(
        "实践教学案例设计专家", "生成实操案例。",
        profile,
        state.get("course_context", ""),
        state.get("resource_plan", ""),
        state["messages"][-1].content,
        "## 💻 4. 实操案例（案例名称+问题描述+需求分析+完整代码+运行结果+扩展思考）\n"
        "代码必须完整可运行。",
    )

    answer = ""
    for evt in call_llm_stream([{"role": "user", "content": prompt}]):
        if evt[0] == "chunk":
            answer += evt[1]
        yield evt
    answer = output_safety_filter(answer)
    yield ("_result", {"messages": [AIMessage(content=answer)]})


def resource_merger_agent(state: AgentState, events_out: List):
    """资源合并智能体（非流式）：整合各智能体输出，检查完整性。"""
    parts = []
    msg_start = state.get("_resource_msg_index", 0)
    for msg in reversed(state["messages"][msg_start:]):
        if isinstance(msg, AIMessage):
            if any(h in msg.content for h in RESOURCE_HEADERS):
                parts.append(msg.content)
                if len(parts) >= 4:
                    break
    combined = "\n\n".join(reversed(parts))

    events_out.append(("status", "✅ 资源合并完成！"))

    missing = [h for h in RESOURCE_HEADERS if h not in combined]
    if missing:
        events_out.append(("status", f"⚠️ 补充生成: {missing}"))
        sup = call_llm_sync([{"role": "user", "content": f"请补充以下资源：{missing}"}])
        combined += "\n\n--- 补充 ---\n" + sup

    return {"messages": [AIMessage(content=combined)]}


def plan_node(state: AgentState, events_out: List):
    """路径规划智能体（非流式）。"""
    profile = state.get("profile") or {}
    course_ctx = state.get("course_context", "")

    events_out.append(("status", "🗺️ 路径规划智能体：正在生成个性化学习路径..."))
    prompt = build_plan_prompt(profile, course_ctx)
    plan_text = call_llm_sync([{"role": "user", "content": prompt}])

    parsed = extract_json(plan_text)
    if parsed and isinstance(parsed, dict) and "steps" in parsed:
        steps = parsed["steps"]
        summary = parsed.get("summary", "个性化学习路径")
    else:
        # 回退到文本行解析
        steps = []
        summary = "个性化学习路径"
        for line in plan_text.split("\n"):
            line = line.strip()
            if line and len(line) > 2 and line[0].isdigit() and "." in line:
                s = line.split(".", 1)[1].strip()
                if s:
                    steps.append({"title": s[:20], "description": s, "goal": "掌握相关知识", "resource_types": ["文档", "练习"]})

    if not steps:
        weak = ", ".join(profile.get("weak_points", [])) or "无"
        steps = [
            {"title": "夯实基础", "description": f"复习核心概念和基本原理，重点关注{weak}", "goal": "建立知识框架", "resource_types": ["文档", "视频"]},
            {"title": "专项练习", "description": "针对薄弱点进行针对性练习和巩固", "goal": "攻克薄弱环节", "resource_types": ["练习", "案例"]},
            {"title": "综合实践", "description": "完成综合案例，将知识融会贯通", "goal": "综合运用", "resource_types": ["案例", "练习"]},
        ]

    # 设置第一步为 current，其余为 todo
    for i, step in enumerate(steps):
        step["status"] = "current" if i == 0 else "todo"

    events_out.append(("status", f"✅ 已生成 {len(steps)} 个学习步骤"))
    events_out.append(("data", json.dumps({"type": "plan", "data": {"steps": steps, "summary": summary}})))
    return {"learning_plan": steps}


# ================== 主流程生成器 ==================

def run_graph(state: AgentState):
    """运行整个多智能体工作流，实时 yield 事件。"""
    events_buf: List = []

    # 阶段1：画像分析（非流式）
    pr = profile_agent(state, events_buf)
    for e in events_buf:
        yield e
    state.update(pr)

    # 阶段2：意图识别（非流式）
    events_buf.clear()
    ir = classify_intent(state, events_buf)
    for e in events_buf:
        yield e
    state.update(ir)

    intent = state["user_intent"]

    # 阶段3：按意图路由
    if intent == "greeting":
        for evt in greeting_node_gen(state):
            if evt[0] == "_result":
                state.update(evt[1])
            elif evt[0] == "_llm_done":
                pass
            else:
                yield evt

    elif intent == "resource":
        # 资源规划（非流式）
        events_buf.clear()
        rp = resource_planner_agent(state, events_buf)
        for e in events_buf:
            yield e
        state.update(rp)

        # 内容生成（流式）
        for evt in content_generator_gen(state):
            if evt[0] == "_result":
                state["messages"].append(evt[1]["messages"][0])
            elif evt[0] == "_llm_done":
                pass
            else:
                yield evt

        # 多模态设计（流式）
        for evt in multimodal_generator_gen(state):
            if evt[0] == "_result":
                state["messages"].append(evt[1]["messages"][0])
            elif evt[0] == "_llm_done":
                pass
            else:
                yield evt

        # 案例生成（流式）
        for evt in case_generator_gen(state):
            if evt[0] == "_result":
                state["messages"].append(evt[1]["messages"][0])
            elif evt[0] == "_llm_done":
                pass
            else:
                yield evt

        # 合并（非流式）
        events_buf.clear()
        mr = resource_merger_agent(state, events_buf)
        for e in events_buf:
            yield e
        if "messages" in mr and mr["messages"]:
            state["messages"].append(mr["messages"][0])

        # 路径规划（非流式）
        events_buf.clear()
        pl = plan_node(state, events_buf)
        for e in events_buf:
            yield e
        state.update(pl)

    elif intent == "tutor":
        for evt in tutor_node_gen(state):
            if evt[0] == "_result":
                state.update(evt[1])
            elif evt[0] == "_llm_done":
                pass
            else:
                yield evt

    elif intent == "evaluation":
        for evt in evaluation_node_gen(state):
            if evt[0] == "_result":
                state.update(evt[1])
            elif evt[0] == "_llm_done":
                pass
            else:
                yield evt

    else:  # chat
        for evt in chat_node_gen(state):
            if evt[0] == "_result":
                state.update(evt[1])
            elif evt[0] == "_llm_done":
                pass
            else:
                yield evt

    yield ("_done", "")


# ================== 会话管理 ==================

sessions: Dict[str, AgentState] = {}
_session_times: Dict[str, float] = {}
_session_lock = threading.Lock()

MAX_SESSIONS = 50
MAX_HUMAN_MESSAGES = 50
TRIM_KEEP = 30


def _evict_oldest_session():
    """LRU 风格驱逐最不活跃的会话（需持有 _session_lock）。"""
    if len(sessions) > MAX_SESSIONS and _session_times:
        oldest = min(_session_times, key=lambda k: _session_times.get(k, 0))
        if oldest in sessions:
            del sessions[oldest]
            del _session_times[oldest]
            logger.debug("LRU 驱逐会话: %s", oldest)


def _trim_messages(state: AgentState):
    """消息历史裁剪：超过阈值时保留最近的。"""
    human_count = sum(1 for m in state["messages"] if isinstance(m, HumanMessage))
    if human_count >= MAX_HUMAN_MESSAGES:
        state["messages"] = state["messages"][-TRIM_KEEP:]
        logger.debug("消息历史已裁剪，保留最近 %d 条", TRIM_KEEP)


# ================== 对外接口 ==================

def process_message(session_id: str, user_message: str):
    """处理用户消息，实时 yield 流式事件供 SSE 推送。"""
    with _session_lock:
        if session_id not in sessions:
            sessions[session_id] = {
                "messages": [],
                "user_intent": "",
                "profile": None,
                "learning_plan": None,
                "course_context": "",
                "resource_plan": "",
            }
        _session_times[session_id] = time.time()
        _evict_oldest_session()
        state = sessions[session_id]

    # 输入安全过滤
    hit = content_safety_check(user_message)
    if hit:
        yield ("status", hit)
        yield ("end", "")
        return

    # 课程上下文匹配
    course_ctx = get_course_context(user_message)
    if course_ctx:
        state["course_context"] = course_ctx

    _trim_messages(state)
    state["messages"].append(HumanMessage(content=user_message))

    try:
        for evt in run_graph(state):
            if evt[0] == "_done":
                break
            yield evt
        yield ("end", "")
    except Exception as e:
        # 异常时回滚已添加的 HumanMessage
        if (state["messages"]
                and isinstance(state["messages"][-1], HumanMessage)
                and state["messages"][-1].content == user_message):
            state["messages"].pop()
        logger.error("处理消息时出错: %s", e)
        yield ("status", f"❌ 处理出错：{e}")
        yield ("error", str(e))
