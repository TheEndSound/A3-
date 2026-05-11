#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
控制台版多智能体学习系统
基于 LangGraph StateGraph 的标准图工作流
"""

import json
from typing import Dict, Any, List, Optional

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from core import (
    load_course_knowledge_base,
    get_course_context,
    extract_json,
    content_safety_check,
    output_safety_filter,
    call_llm_console,
    call_llm_sync,
    search_bilibili_videos,
    build_profile_prompt,
    build_intent_prompt,
    build_greeting_prompt,
    build_chat_system_prompt,
    build_tutor_prompt,
    build_evaluation_prompt,
    build_plan_prompt,
    get_style_guide,
    is_tutor_request,
    is_identity_question,
    merge_profile,
    AgentState,
    RESOURCE_HEADERS,
    logger,
)


# ================== 智能体节点 ==================

def profile_agent(state: AgentState) -> Dict[str, Any]:
    """画像分析智能体：从对话中提取 6 维学生画像。"""
    existing_profile = state.get("profile") or {}
    recent_msgs = state["messages"][-4:]
    context = "\n".join(
        [f"{'用户' if isinstance(m, HumanMessage) else 'AI'}: {m.content}" for m in recent_msgs]
    )

    safety_hit = content_safety_check(context)
    if safety_hit:
        logger.warning("安全过滤命中: %s", safety_hit)
        return {"profile": existing_profile}

    prompt = build_profile_prompt(existing_profile, context)
    resp = call_llm_sync([{"role": "user", "content": prompt}])
    new_profile = extract_json(resp)

    if new_profile is not None:
        merged = merge_profile(existing_profile, new_profile)
        logger.info("画像更新成功: %s", json.dumps(merged, ensure_ascii=False))
    else:
        logger.info("画像解析失败，保留旧画像。原始返回前200字: %s", resp[:200])
        merged = existing_profile
    return {"profile": merged}


def classify_intent(state: AgentState) -> Dict[str, Any]:
    """意图识别智能体：将用户输入分类为 5 种意图之一。"""
    last_msg = state["messages"][-1].content
    prompt = build_intent_prompt(last_msg)
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
    logger.info("意图识别: %s", intent)
    return {"user_intent": intent}


def greeting_node(state: AgentState) -> Dict[str, Any]:
    """问候节点：友好回应问候或身份询问。"""
    user_msg = state["messages"][-1].content if state["messages"] else ""
    prompt = build_greeting_prompt(user_msg)
    print("\nAI: ", end="")
    answer = call_llm_console([{"role": "user", "content": prompt}])
    return {"messages": [AIMessage(content=answer)]}


def chat_node(state: AgentState) -> Dict[str, Any]:
    """普通对话节点：根据画像和课程上下文回答学术问题。"""
    profile = state.get("profile") or {}
    course_ctx = state.get("course_context", "")

    history = []
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage):
            history.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            history.append({"role": "assistant", "content": msg.content})

    system_prompt = build_chat_system_prompt(profile, course_ctx)

    last_question = history[-1]["content"] if history else ""
    if is_identity_question(last_question):
        identity_msg = {
            "role": "user",
            "content": "【系统指令】你的名字是「AI智学」，是一个智能学习助手。请直接告诉用户你的名字「AI智学」和你提供的功能。",
        }
        messages = [{"role": "system", "content": system_prompt}, identity_msg] + history
    else:
        messages = [{"role": "system", "content": system_prompt}] + history

    print("AI: ", end="")
    answer = call_llm_console(messages)
    answer = output_safety_filter(answer)

    # 如果有历史学习计划，追加推荐路径
    plan = state.get("learning_plan")
    if plan:
        answer += "\n\n📌 **根据你的学习情况，推荐学习路径：**\n" + "\n".join(
            f"{i + 1}. {step}" for i, step in enumerate(plan)
        )
    return {"messages": [AIMessage(content=answer)]}


def tutor_node(state: AgentState) -> Dict[str, Any]:
    """智能辅导节点：为学生的疑问提供多角度答疑。"""
    last_question = state["messages"][-1].content
    profile = state.get("profile") or {}
    course_ctx = state.get("course_context", "")

    prompt = build_tutor_prompt(last_question, profile, course_ctx)
    print("\n🧑‍🏫 [智能辅导节点] 正在为你详细解答...\n")
    answer = call_llm_console([{"role": "user", "content": prompt}])
    answer = output_safety_filter(answer)
    return {"messages": [AIMessage(content=answer)]}


def evaluation_node(state: AgentState) -> Dict[str, Any]:
    """学习效果评估节点：分析对话历史，生成评估报告。"""
    profile = state.get("profile") or {}

    history_text = ""
    for msg in state["messages"][-10:]:
        role = "用户" if isinstance(msg, HumanMessage) else "AI"
        history_text += f"{role}: {msg.content[:200]}\n"

    prompt = build_evaluation_prompt(profile, history_text)
    print("\n📊 [学习效果评估节点] 正在生成评估报告...\n")
    answer = call_llm_console([{"role": "user", "content": prompt}])
    answer = output_safety_filter(answer)

    # 尝试提取结构化评估数据并打印
    eval_data = extract_json(answer)
    if eval_data and "overall_score" in eval_data:
        print("=" * 50)
        print("📊 学习效果评估结果")
        print("=" * 50)
        print(f"  综合分数：{eval_data.get('overall_score', '--')}/100")
        print(f"  知识掌握度：{eval_data.get('knowledge_level', '--')}")
        print(f"  学习效率：{eval_data.get('efficiency_level', '--')}")
        weak_pts = eval_data.get('weak_points_list', [])
        print(f"  薄弱点：{', '.join(weak_pts) if weak_pts else '无'}")
        print(f"  进步情况：{eval_data.get('progress_summary', '--')}")
        print(f"  学习建议：{eval_data.get('suggestions', '--')}")
        print(f"  节奏建议：{eval_data.get('pace_recommendation', '--')}")
        print("=" * 50)

    return {"messages": [AIMessage(content=answer)]}


# ================== 资源生成多智能体协作组 ==================

def resource_planner_agent(state: AgentState) -> Dict[str, Any]:
    """资源规划智能体：分析画像和课程知识，规划资源内容方向。"""
    profile = state.get("profile") or {}
    last_question = state["messages"][-1].content
    course_ctx = state.get("course_context", "")

    print("\n📋 [资源规划智能体] 正在分析需求并规划资源内容...\n")
    prompt = f"""你是一个学习资源规划专家。根据学生画像和课程上下文，规划要生成的5种学习资源的具体内容方向。

学生画像：
- 知识基础：{profile.get('knowledge_base', '初级')}
- 学习风格：{profile.get('learning_style', '视觉型')}
- 薄弱点：{', '.join(profile.get('weak_points', ['无']))}
- 兴趣方向：{profile.get('interest', '编程')}

{course_ctx}
学生提问：{last_question}

请输出一份资源规划方案（JSON格式），包含以下字段：
- "topic": 本次资源生成的核心主题
- "teaching_approach": 教学方法建议
- "focus_areas": 重点讲解的知识点列表
- "difficulty_level": 难度级别
- "resource_order": 推荐资源生成顺序
- "style_recommendation": 根据学习风格的内容呈现建议

只输出JSON对象，不要有其他解释。"""

    resp = call_llm_sync([{"role": "user", "content": prompt}])
    plan = extract_json(resp)
    if plan:
        logger.info("资源规划完成: 主题=%s | 难度=%s", plan.get("topic", "未指定"), plan.get("difficulty_level", "中级"))
        return {"resource_plan": json.dumps(plan, ensure_ascii=False), "_resource_msg_index": len(state["messages"])}
    else:
        logger.info("资源规划使用默认方案")
        return {"resource_plan": resp, "_resource_msg_index": len(state["messages"])}


def _build_resource_prompt(agent_title: str, task_description: str, profile: dict, course_ctx: str, resource_plan: str, last_question: str, output_format: str) -> str:
    """构建资源生成类智能体的 prompt。"""
    return f"""你是一个{agent_title}。{task_description}

资源规划参考：{resource_plan}
{course_ctx}
学生画像：
- 知识基础：{profile.get('knowledge_base', '初级')}
- 学习风格：{profile.get('learning_style', '视觉型')}
- 薄弱点：{', '.join(profile.get('weak_points', ['无']))}
- 兴趣方向：{profile.get('interest', '编程')}

学生提问：{last_question}

请严格按照以下格式输出：
{output_format}"""


def content_generator_agent(state: AgentState) -> Dict[str, Any]:
    """内容生成智能体：生成讲解文档和练习题。"""
    profile = state.get("profile") or {}

    print("\n📝 [内容生成智能体] 正在生成讲解文档和练习题...\n")
    prompt = _build_resource_prompt(
        "教学内容生成专家",
        "请生成讲解文档和练习题两种资源。",
        profile,
        state.get("course_context", ""),
        state.get("resource_plan", ""),
        state["messages"][-1].content,
        """## 📘 1. 讲解文档
（详细文档：概念解释、核心原理、代码示例、复杂度分析、应用场景。针对薄弱点重点讲解。）

## 📝 2. 练习题
（至少3道：选择题2道+编程题1-2道，附答案和解析）""",
    )

    answer = call_llm_console([{"role": "user", "content": prompt}])
    answer = output_safety_filter(answer)
    return {"messages": [AIMessage(content=answer)]}


def multimodal_generator_agent(state: AgentState) -> Dict[str, Any]:
    """多模态设计智能体：生成思维导图和视频脚本。"""
    profile = state.get("profile") or {}

    print("\n🎨 [多模态设计智能体] 正在生成思维导图和视频脚本...\n")
    prompt = _build_resource_prompt(
        "多模态教学内容设计专家",
        "请生成知识点思维导图（Mermaid格式）和视频脚本。",
        profile,
        state.get("course_context", ""),
        state.get("resource_plan", ""),
        state["messages"][-1].content,
        """## 🗺️ 3. 知识点思维导图
输出 Mermaid mindmap 代码块（必须用 ```mermaid 包裹），格式示例：
```mermaid
mindmap
  root((核心主题))
    分支1
      子节点A
      子节点B
    分支2
      子节点C
      子节点D
```
要求：至少3个一级分支，每分支至少2个子节点，根节点用 ((双括号)) 包裹。

## 🎥 5. 多模态教学视频/动画脚本
（3-5个分镜场景，每个场景含：画面描述、旁白文本、动画效果建议、时长）""",
    )

    answer = call_llm_console([{"role": "user", "content": prompt}])
    answer = output_safety_filter(answer)

    # 搜索B站相关视频
    try:
        rp_raw = state.get("resource_plan", "")
        plan = extract_json(rp_raw) if rp_raw else {}
        topic = plan.get("topic", "") if plan else ""
        if topic:
            videos = search_bilibili_videos(topic)
            if videos:
                print("\n🎬 B站推荐视频：")
                for v in videos:
                    print(f"  - {v['title']}  by {v['author']} ({v['play']} 播放)")
                    print(f"    https://www.bilibili.com/video/{v['bvid']}")
    except Exception:
        pass

    return {"messages": [AIMessage(content=answer)]}


def case_generator_agent(state: AgentState) -> Dict[str, Any]:
    """案例生成智能体：生成实操案例。"""
    profile = state.get("profile") or {}

    print("\n💻 [案例生成智能体] 正在生成实操案例...\n")
    prompt = _build_resource_prompt(
        "实践教学案例设计专家",
        "请生成实操案例。",
        profile,
        state.get("course_context", ""),
        state.get("resource_plan", ""),
        state["messages"][-1].content,
        """## 💻 4. 实操案例
（完整案例：案例名称、问题描述、需求分析、完整代码（带注释）、运行结果、扩展思考）""",
    )

    answer = call_llm_console([{"role": "user", "content": prompt}])
    answer = output_safety_filter(answer)
    return {"messages": [AIMessage(content=answer)]}


def resource_merger_agent(state: AgentState) -> Dict[str, Any]:
    """资源合并智能体：整合各智能体输出，检查完整性并补充缺失资源。"""
    resource_parts = []
    msg_start = state.get("_resource_msg_index", 0)
    for msg in reversed(state["messages"][msg_start:]):
        if isinstance(msg, AIMessage):
            content = msg.content
            if any(h in content for h in RESOURCE_HEADERS):
                resource_parts.append(content)
                if len(resource_parts) >= 4:
                    break
    combined = "\n\n".join(reversed(resource_parts))

    print("\n✅ [资源合并] 正在整合各智能体生成的资源...")
    missing = [h for h in RESOURCE_HEADERS if h not in combined]
    if missing:
        logger.info("资源校验：缺少 %s，补充生成...", missing)
        supplement = call_llm_sync([{"role": "user", "content": f"请补充以下资源：{missing}"}])
        combined += "\n\n--- 补充 ---\n" + supplement

    return {"messages": [AIMessage(content=combined)]}


def plan_node(state: AgentState) -> Dict[str, Any]:
    """路径规划智能体：基于画像和课程知识库生成个性化学习路径。"""
    profile = state.get("profile") or {}
    course_ctx = state.get("course_context", "")

    prompt = build_plan_prompt(profile, course_ctx)
    plan_text = call_llm_sync([{"role": "user", "content": prompt}])

    parsed = extract_json(plan_text)
    if parsed and isinstance(parsed, dict) and "steps" in parsed:
        steps = [s.get("title", s.get("description", str(s)))[:50] for s in parsed["steps"]]
    else:
        # 回退到文本行解析
        steps = []
        for line in plan_text.split("\n"):
            line = line.strip()
            if line and len(line) > 2 and line[0].isdigit() and "." in line:
                step_content = line.split(".", 1)[1].strip()
                if step_content:
                    steps.append(step_content)

    if not steps:
        weak = ", ".join(profile.get("weak_points", [])) or "无"
        steps = [
            f"夯实基础 - 复习相关概念 | 目标：建立框架 | 资源：讲解文档",
            f"针对性练习 - 针对{weak}专项练习 | 目标：攻克薄弱 | 资源：练习题",
            f"综合实践 - 完成项目案例 | 目标：综合运用 | 资源：实操案例",
            f"巩固提升 - 梳理知识体系 | 目标：查漏补缺 | 资源：思维导图+视频",
        ]

    logger.info("学习路径已生成: %d 个步骤", len(steps))
    return {"learning_plan": steps}


# ================== 路由逻辑 ==================

def route_after_classification(state: AgentState) -> str:
    intent = state["user_intent"]
    if intent == "greeting":
        return "greeting_node"
    elif intent == "resource":
        return "resource_planner_agent"
    else:
        last_msg = state["messages"][-1].content if state["messages"] else ""
        if "评估" in last_msg and ("学习" in last_msg or "效果" in last_msg):
            return "evaluation_node"
        elif is_tutor_request(last_msg):
            return "tutor_node"
        return "chat_node"


# ================== 构建 LangGraph 工作流 ==================

def build_graph():
    workflow = StateGraph(AgentState)

    # 基础节点
    workflow.add_node("profile_agent", profile_agent)
    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("greeting_node", greeting_node)
    workflow.add_node("chat_node", chat_node)
    workflow.add_node("tutor_node", tutor_node)
    workflow.add_node("evaluation_node", evaluation_node)

    # 资源生成多智能体协作组
    workflow.add_node("resource_planner_agent", resource_planner_agent)
    workflow.add_node("content_generator_agent", content_generator_agent)
    workflow.add_node("multimodal_generator_agent", multimodal_generator_agent)
    workflow.add_node("case_generator_agent", case_generator_agent)
    workflow.add_node("resource_merger_agent", resource_merger_agent)
    workflow.add_node("plan_node", plan_node)

    # 图结构
    workflow.set_entry_point("profile_agent")
    workflow.add_edge("profile_agent", "classify_intent")
    workflow.add_conditional_edges(
        "classify_intent",
        route_after_classification,
        {
            "greeting_node": "greeting_node",
            "chat_node": "chat_node",
            "tutor_node": "tutor_node",
            "evaluation_node": "evaluation_node",
            "resource_planner_agent": "resource_planner_agent",
        },
    )

    # 资源生成链路
    workflow.add_edge("resource_planner_agent", "content_generator_agent")
    workflow.add_edge("content_generator_agent", "multimodal_generator_agent")
    workflow.add_edge("multimodal_generator_agent", "case_generator_agent")
    workflow.add_edge("case_generator_agent", "resource_merger_agent")
    workflow.add_edge("resource_merger_agent", "plan_node")
    workflow.add_edge("plan_node", END)

    # 各叶子节点 → END
    workflow.add_edge("greeting_node", END)
    workflow.add_edge("chat_node", END)
    workflow.add_edge("tutor_node", END)
    workflow.add_edge("evaluation_node", END)

    return workflow.compile()


# ================== 交互运行 ==================

def run_interactive(app):
    print("🚀 个性化学习多智能体系统已启动（5种资源生成 + 学习路径规划 + 智能辅导 + 效果评估）")
    print("支持功能：")
    print("  - 输入任意内容开始对话，系统会自动分析画像")
    print("  - 输入'生成资源'或'帮我练习'触发5种资源生成+路径规划")
    print("  - 输入'评估我的学习效果'触发学习效果评估")
    print("  - 输入'为什么...'或'我不懂...'触发智能辅导")
    print("  - 输入 'exit' 退出程序\n")

    load_course_knowledge_base()

    state = {
        "messages": [],
        "user_intent": "",
        "profile": None,
        "learning_plan": None,
        "course_context": "",
        "resource_plan": "",
    }

    while True:
        user_input = input("\n👤 你: ")
        if user_input.lower() == "exit":
            break

        course_ctx = get_course_context(user_input)
        if course_ctx:
            state["course_context"] = course_ctx

        state["messages"].append(HumanMessage(content=user_input))
        final_state = app.invoke(state)

        state["messages"] = final_state.get("messages", state["messages"])
        state["user_intent"] = final_state.get("user_intent", "")
        state["profile"] = final_state.get("profile", state["profile"])
        state["learning_plan"] = final_state.get("learning_plan", state["learning_plan"])
        state["resource_plan"] = final_state.get("resource_plan", state["resource_plan"])
        state["course_context"] = final_state.get("course_context", state["course_context"])


if __name__ == "__main__":
    app = build_graph()
    run_interactive(app)
