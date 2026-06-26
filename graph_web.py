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
    get_user_documents_context,
    extract_json,
    content_safety_check,
    output_safety_filter,
    call_llm_stream,
    call_llm_sync,
    call_doubao_image,
    search_bilibili_videos,
    search_web,
    enrich_search_results,
    format_search_context,
    build_profile_prompt,
    build_intent_prompt,
    build_greeting_prompt,
    build_chat_system_prompt,
    build_tutor_prompt,
    build_evaluation_prompt,
    build_plan_prompt,
    build_single_resource_prompt,
    is_tutor_request,
    is_identity_question,
    merge_profile,
    AgentState,
    RESOURCE_HEADERS,
    logger,
    _has_time_reference,
)
from database import (
    upsert_session as db_upsert_session,
    insert_message as db_insert_message,
    update_session_preview as db_update_session_preview,
    get_session_messages as db_get_session_messages,
    save_profile as db_save_profile,
    save_plan as db_save_plan,
    get_plan as db_get_plan,
)


# ================== 智能体节点（生成器版） ==================

# 深度思考模式 — 高优先级格式指令（直接拼入 prompt 开头，确保不被忽略）
_DEEP_THINKING_META = (
    "【深度思考模式 — 必须严格遵守】\n"
    "你的回复必须按以下格式输出，否则视为无效：\n\n"
    "<深度思考>\n"
    "（完整推理：问题拆解→关键概念→多角度分析→推导论证→反思验证）\n"
    "</深度思考>\n\n"
    "（最终答案 — 可包含任意格式：标题、列表、代码、表格等）\n\n"
    "硬性规则：\n"
    "1. <深度思考> 和 </深度思考> 标签必须成对出现，缺一不可\n"
    "2. 思考过程写在标签内，最终答案写在标签外\n"
    "3. 只有一对标签，不要嵌套\n"
)

def profile_agent(state: AgentState, events_out: List):
    """画像分析智能体（非流式）—— 深度分析最近对话，抽取10维度画像。

    改进：分析最近 12 条消息（vs 旧版 4 条），覆盖更多对话上下文，
    能更准确推断学生的专业背景、学习目标、认知风格等深层特征。
    """
    existing = state.get("profile") or {}
    # 取更多历史消息以捕获深层特征
    recent = state["messages"][-12:]
    # 截断过长的消息内容，避免超出 token 限制
    ctx_lines = []
    for m in recent:
        role = '用户' if isinstance(m, HumanMessage) else 'AI'
        content = m.content[:300]  # 每条截断到300字
        ctx_lines.append(f"{role}: {content}")
    ctx = "\n".join(ctx_lines)

    hit = content_safety_check(ctx)
    if hit:
        events_out.append(("status", hit))
        return {"profile": existing}

    prompt = build_profile_prompt(existing, ctx)
    resp = call_llm_sync([{"role": "user", "content": prompt}])
    new_p = extract_json(resp)

    if new_p is not None:
        merged = merge_profile(existing, new_p)
        # 更丰富状态信息
        dims = []
        if merged.get("knowledge_base"): dims.append(merged["knowledge_base"] + "基础")
        if merged.get("cognitive_style"): dims.append(merged["cognitive_style"])
        if merged.get("motivation_level"): dims.append("动机" + merged["motivation_level"])
        status_msg = "✅ 画像已更新：" + " | ".join(dims) if dims else "✅ 画像已更新"
        events_out.append(("status", status_msg))
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

    # 深度思考：将格式指令直接拼入系统提示词开头（合并为一条消息，优先级更高）
    if state.get("deep_thinking"):
        msgs[0]["content"] = _DEEP_THINKING_META + "\n\n" + msgs[0]["content"]

    answer = ""
    for evt in call_llm_stream(msgs):
        if evt[0] == "chunk":
            answer += evt[1]
        yield evt

    answer = output_safety_filter(answer)

    # 追加学习路径推荐（新闻模式 / 搜索增强模式下跳过，用户处于信息获取模式）
    plan = state.get("learning_plan")
    has_search = "[联网搜索结果" in course_ctx
    if plan and not has_search:
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

    msgs = [{"role": "user", "content": prompt}]
    if state.get("deep_thinking"):
        msgs[0]["content"] = _DEEP_THINKING_META + "\n\n" + msgs[0]["content"]

    answer = ""
    for evt in call_llm_stream(msgs):
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

    msgs = [{"role": "user", "content": prompt}]
    if state.get("deep_thinking"):
        msgs[0]["content"] = _DEEP_THINKING_META + "\n\n" + msgs[0]["content"]

    answer = ""
    for evt in call_llm_stream(msgs):
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
        "⛔ 严禁使用 graph/flowchart 语法（如 A --> B、subgraph 等），只能使用 mindmap 语法。\n"
        "⛔ 严禁在 mindmap 中使用方括号 [ ]、圆括号 ( ) 包裹节点文字，节点文字必须是纯文本，不需要任何括号包裹。\n"
        "⛔ 严禁使用 --> 箭头符号。\n"
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

    # 提取诊断
    diagnosis = parsed.get("diagnosis", "") if parsed and isinstance(parsed, dict) else ""

    events_out.append(("status", f"✅ 已生成 {len(steps)} 个学习步骤"))
    events_out.append(("data", json.dumps({"type": "plan", "data": {"steps": steps, "summary": summary, "diagnosis": diagnosis}})))
    return {"learning_plan": steps, "_plan_diagnosis": diagnosis, "_plan_summary": summary}


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

    # 新闻/资讯模式：强制走 chat 通道，阻止资源生成管线
    course_ctx = state.get("course_context", "")
    if "信息摘要助手" in course_ctx and intent == "resource":
        intent = "chat"
        state["user_intent"] = "chat"
        yield ("status", "📰 检测到资讯查询，切换为信息摘要模式")

    # 阶段3：按意图路由
    if intent == "greeting":
        for evt in greeting_node_gen(state):
            if evt[0] == "_result":
                result_data = evt[1]
                if "messages" in result_data:
                    state["messages"].extend(result_data["messages"])
                for k, v in result_data.items():
                    if k != "messages":
                        state[k] = v
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

    elif intent == "tutor":
        for evt in tutor_node_gen(state):
            if evt[0] == "_result":
                result_data = evt[1]
                if "messages" in result_data:
                    state["messages"].extend(result_data["messages"])
                for k, v in result_data.items():
                    if k != "messages":
                        state[k] = v
            elif evt[0] == "_llm_done":
                pass
            else:
                yield evt

    elif intent == "evaluation":
        for evt in evaluation_node_gen(state):
            if evt[0] == "_result":
                result_data = evt[1]
                if "messages" in result_data:
                    state["messages"].extend(result_data["messages"])
                for k, v in result_data.items():
                    if k != "messages":
                        state[k] = v
            elif evt[0] == "_llm_done":
                pass
            else:
                yield evt

    else:  # chat
        for evt in chat_node_gen(state):
            if evt[0] == "_result":
                result_data = evt[1]
                if "messages" in result_data:
                    state["messages"].extend(result_data["messages"])
                for k, v in result_data.items():
                    if k != "messages":
                        state[k] = v
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

def _image_gen_node_gen(user_message: str):
    """Image generation fast path: call doubao Seedream API directly."""
    yield ("status", "🎨 正在生成图像...")
    result = call_doubao_image(user_message)
    if result["success"] and result["url"]:
        yield ("image_url", result["url"])
        yield ("status", "✅ 图像生成完成")
        yield ("_result", {"messages": [AIMessage(content=f"[生成图像]\n![AI生成图片]({result['url']})")]})
    else:
        err_msg = result.get("error", "未知错误")
        yield ("error", f"图像生成失败: {err_msg}")
        yield ("_result", {"messages": [AIMessage(content=f"[图像生成失败] {err_msg}")]})


# ---------- 单资源输出净化 ----------

_SANITIZE_DOC_BAN = [
    r"##\s*选择题[\s\S]*?(?=##\s|\Z)",
    r"##\s*编程题[\s\S]*?(?=##\s|\Z)",
    r"##\s*练习题[\s\S]*?(?=##\s|\Z)",
    r"##\s*难度分级[\s\S]*?(?=##\s|\Z)",
    r"##\s*课后作业[\s\S]*?(?=##\s|\Z)",
]
_SANITIZE_EXERCISE_BAN = [
    r"##\s*概念解释[\s\S]*?(?=##\s|\Z)",
    r"##\s*原理说明[\s\S]*?(?=##\s|\Z)",
    r"##\s*知识回顾[\s\S]*?(?=##\s|\Z)",
    r"##\s*复杂度分析[\s\S]*?(?=##\s|\Z)",
    r"##\s*应用场景[\s\S]*?(?=##\s|\Z)",
    r"##\s*背景知识[\s\S]*?(?=##\s|\Z)",
    r"##\s*学习目标[\s\S]*?(?=##\s|\Z)",
]


def _sanitize_resource_output(text: str, resource_type: str) -> str:
    """后处理净化：从输出中剥离不属于当前资源类型的章节。"""
    import re as _re
    if resource_type == "doc":
        for pattern in _SANITIZE_DOC_BAN:
            text = _re.sub(pattern, "", text)
        # 去掉常见的 LLM 问候/结尾语
        text = _re.sub(r"^(好的|没问题|明白了|收到)[，,].*?\n", "", text)
        text = _re.sub(r"\n*希望.*?(有帮助|能帮到你).*?[。！]\s*$", "", text)
    elif resource_type == "exercise":
        for pattern in _SANITIZE_EXERCISE_BAN:
            text = _re.sub(pattern, "", text)
        # 去掉 LLM 常见的讲解性开头
        text = _re.sub(r"^(好的|没问题|明白了|收到)[，,].*?\n", "", text)
        # 去掉"同学你好"等开场白
        text = _re.sub(r"^.{0,30}同学.{0,50}\n", "", text)
        text = _re.sub(r"^\*\*\[.+\]\*\*\s*\n*", "", text)
        text = _re.sub(r"\n*希望.*?(有帮助|能帮到你|练习顺利).*?[！。]\s*$", "", text)
    return text.strip()


def process_message(session_id: str, user_message: str, image_mode: bool = False, web_search: bool = False, deep_thinking: bool = False, resource_type: str = ""):
    """处理用户消息，实时 yield 流式事件供 SSE 推送。
    resource_type: 单资源快速通道 — doc/exercise/mindmap/video/case/ppt，跳过智能体管线直接生成。"""
    logger.info("process_message: session=%s, image_mode=%s, web_search=%s, deep_thinking=%s, resource_type=%s, msg=%s",
                 session_id[:12], image_mode, web_search, deep_thinking, resource_type, user_message[:60])
    with _session_lock:
        if session_id not in sessions:
            # 尝试从数据库恢复会话历史
            db_msgs = db_get_session_messages(session_id)
            # 恢复学习路径
            db_plan = db_get_plan(session_id)
            restored_plan = db_plan.get("steps", []) if db_plan else None
            if db_msgs:
                msgs = []
                # 加载最近的消息用于对话上下文（最多40条）
                for m in db_msgs[-40:]:
                    if m["role"] == "human":
                        msgs.append(HumanMessage(content=m["content"]))
                    else:
                        msgs.append(AIMessage(content=m["content"]))
                sessions[session_id] = {
                    "messages": msgs,
                    "user_intent": "",
                    "profile": None,
                    "learning_plan": restored_plan,
                    "course_context": "",
                    "resource_plan": "",
                    "deep_thinking": False,
                    "_next_seq": len(db_msgs),
                }
            else:
                sessions[session_id] = {
                    "messages": [],
                    "user_intent": "",
                    "profile": None,
                    "learning_plan": restored_plan if restored_plan else None,
                    "course_context": "",
                    "resource_plan": "",
                    "deep_thinking": False,
                    "_next_seq": 0,
                }
        _session_times[session_id] = time.time()
        _evict_oldest_session()
        state = sessions[session_id]
        if "_next_seq" not in state:
            state["_next_seq"] = len(db_get_session_messages(session_id))

    # 图像模式：快速通道，跳过所有智能体
    if image_mode:
        # 保存用户消息到数据库
        db_upsert_session(session_id)
        db_insert_message(session_id, "human", user_message, state["_next_seq"])
        db_update_session_preview(session_id, "human", user_message)
        state["_next_seq"] += 1
        state["messages"].append(HumanMessage(content=user_message))
        pre_msg_count = len(state["messages"])

        try:
            for evt in _image_gen_node_gen(user_message):
                if evt[0] == "_result":
                    # 不能直接 state.update() —— 会覆盖 messages 列表导致丢消息
                    # 必须手动 extend messages 以保持 pre_msg_count 切片正确
                    result_data = evt[1]
                    if "messages" in result_data:
                        state["messages"].extend(result_data["messages"])
                    # 更新其他 state 字段
                    for k, v in result_data.items():
                        if k != "messages":
                            state[k] = v
                elif evt[0] == "_llm_done":
                    pass
                else:
                    yield evt

            new_msgs = [m for m in state["messages"][pre_msg_count:] if isinstance(m, AIMessage)]
            for m in new_msgs:
                db_insert_message(session_id, "ai", m.content, state["_next_seq"])
                db_update_session_preview(session_id, "ai", m.content)
                state["_next_seq"] += 1
            yield ("end", "")
        except Exception as e:
            logger.error("图像生成出错: %s", e)
            yield ("status", f"❌ 图像生成出错：{e}")
            yield ("error", str(e))
        return

    # 单资源快速通道：跳过智能体管线，直接聚焦生成
    if resource_type and resource_type in ("doc", "exercise", "mindmap", "video", "case", "ppt"):
        yield ("status", f"🎯 正在生成{resource_type}资源...")
        db_upsert_session(session_id)
        db_insert_message(session_id, "human", user_message, state["_next_seq"])
        db_update_session_preview(session_id, "human", user_message)
        state["_next_seq"] += 1
        state["messages"].append(HumanMessage(content=user_message))
        pre_msg_count = len(state["messages"])

        try:
            sys_prompt, user_prompt = build_single_resource_prompt(user_message, resource_type)
            msgs = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ]
            answer = ""
            for evt in call_llm_stream(msgs, temperature=0):
                if evt[0] == "chunk":
                    answer += evt[1]
                    yield evt
                elif evt[0] not in ("_result", "_llm_done"):
                    yield evt
            answer = output_safety_filter(answer)
            answer = _sanitize_resource_output(answer, resource_type)
            # 直接追加到 state.messages，不 yield _result（AIMessage 不可 JSON 序列化）
            state["messages"].append(AIMessage(content=answer))
            db_insert_message(session_id, "ai", answer, state["_next_seq"])
            db_update_session_preview(session_id, "ai", answer)
            state["_next_seq"] += 1
            yield ("end", "")
        except Exception as e:
            logger.error("单资源生成出错: %s", e)
            yield ("status", f"❌ 生成出错：{e}")
            yield ("error", str(e))
        return

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

    # 加载用户知识库文档并追加到课程上下文
    user_docs_ctx = get_user_documents_context(session_id)
    if user_docs_ctx:
        if state.get("course_context"):
            state["course_context"] += "\n\n" + user_docs_ctx
        else:
            state["course_context"] = user_docs_ctx

    # 联网搜索：调用 open-webSearch 本地服务（多引擎，免费）
    search_context = ""
    if web_search:
        yield ("status", "🌐 正在联网搜索...")
        search_results = search_web(user_message)
        if search_results:
            yield ("status", "📄 正在获取页面内容...")
            enrich_search_results(search_results)
            search_context = format_search_context(search_results, user_message)
            # 注入 course_context（供系统提示词使用）
            if state.get("course_context"):
                state["course_context"] = search_context + "\n" + state["course_context"]
            else:
                state["course_context"] = search_context
            yield ("status", f"🌐 已获取 {len(search_results)} 条搜索结果")
        else:
            yield ("status", "🌐 搜索无结果，请稍后重试")

    # 深度思考模式：存储到 state，由各节点在 LLM 调用时注入高优先级元指令
    state["deep_thinking"] = deep_thinking
    if deep_thinking:
        yield ("status", "🔍 深度思考模式已激活")

    # 保存用户消息到数据库（原始消息，不含搜索上下文）
    db_upsert_session(session_id)
    db_insert_message(session_id, "human", user_message, state["_next_seq"])
    db_update_session_preview(session_id, "human", user_message)
    state["_next_seq"] += 1

    _trim_messages(state)

    # 构建发给 LLM 的消息：如果有搜索结果，直接嵌入用户消息
    # 时间指代词加注具体日期，防止 LLM 误解（如"昨天"→泰剧）
    llm_message = user_message
    if search_context:
        safe_query = user_message
        if _has_time_reference(user_message):
            import datetime
            today = datetime.date.today()
            safe_query = safe_query.replace("前天", f"前天({(today - datetime.timedelta(days=2)).strftime('%Y年%m月%d日')})")
            safe_query = safe_query.replace("昨天", f"昨天({(today - datetime.timedelta(days=1)).strftime('%Y年%m月%d日')})")
            safe_query = safe_query.replace("今天", f"今天({today.strftime('%Y年%m月%d日')})")
            safe_query = safe_query.replace("明天", f"明天({(today + datetime.timedelta(days=1)).strftime('%Y年%m月%d日')})")
            safe_query = safe_query.replace("后天", f"后天({(today + datetime.timedelta(days=2)).strftime('%Y年%m月%d日')})")
        llm_message = f"{search_context}\n---\n用户问题: {safe_query}"
    state["messages"].append(HumanMessage(content=llm_message))

    # 记录执行前的消息数，用于追踪 AI 生成的新消息
    pre_msg_count = len(state["messages"])

    saved_ai_count = 0
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
    finally:
        # 无论正常结束、LLM异常、还是客户端断连，都确保持久化 AI 消息
        new_msgs = [m for m in state["messages"][pre_msg_count:] if isinstance(m, AIMessage)]
        for m in new_msgs:
            try:
                db_insert_message(session_id, "ai", m.content, state["_next_seq"])
                db_update_session_preview(session_id, "ai", m.content)
                state["_next_seq"] += 1
                saved_ai_count += 1
            except Exception as save_err:
                logger.error("保存AI消息失败: %s", save_err)
        # 保存画像
        profile = state.get("profile")
        if profile:
            try:
                db_save_profile(session_id, profile)
            except Exception:
                pass
        # 保存学习路径
        plan_steps = state.get("learning_plan")
        if plan_steps:
            try:
                diagnosis = state.get("_plan_diagnosis", "")
                summary = state.get("_plan_summary", "个性化学习路径")
                db_save_plan(session_id, diagnosis, summary, plan_steps)
            except Exception:
                pass
        if saved_ai_count > 0:
            logger.info("持久化 %d 条AI消息 (session=%s)", saved_ai_count, session_id[:20])
