#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import re
import requests
from typing import TypedDict, Annotated, Dict, Any, List, Optional, Callable
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

# 加载 .env 环境变量
load_dotenv()

# 讯飞星火 API 配置
API_KEY = os.getenv("XF_API_KEY")
if not API_KEY:
    raise ValueError("请在 .env 文件中设置 XF_API_KEY")
URL = "https://spark-api-open.xf-yun.com/v2/chat/completions"


# ================== 加载课程知识库 ==================

_COURSE_KB = None

def load_course_knowledge_base() -> dict:
    global _COURSE_KB
    if _COURSE_KB is not None:
        return _COURSE_KB
    kb_path = os.path.join(os.path.dirname(__file__), "course_knowledge_base.json")
    if os.path.exists(kb_path):
        with open(kb_path, "r", encoding="utf-8") as f:
            _COURSE_KB = json.load(f)
        print(f"[课程知识库] 已加载: {_COURSE_KB['course']['name']} ({len(_COURSE_KB['chapters'])} 章)")
    else:
        _COURSE_KB = {"course": {"name": "默认课程"}, "chapters": [], "knowledge_graph": {}}
    return _COURSE_KB


def find_matching_knowledge(query: str, kb: dict, top_k: int = 3) -> List[dict]:
    query_lower = query.lower()
    matches = []
    for ch in kb.get("chapters", []):
        ch_score = 0
        if ch["title"].lower() in query_lower or any(word in query_lower for word in ch["title"].split()):
            ch_score += 5
        for kp in ch.get("knowledge_points", []):
            score = ch_score
            kp_name = kp["name"].lower()
            if kp_name in query_lower:
                score += 3
            elif any(word in query_lower for word in kp_name.split()):
                score += 1
            if score > 0:
                matches.append({
                    "chapter": ch["title"],
                    "knowledge_point": kp["name"],
                    "difficulty": kp.get("difficulty", "中级"),
                    "score": score
                })
    matches.sort(key=lambda x: x["score"], reverse=True)
    seen = set()
    unique = []
    for m in matches:
        key = m["knowledge_point"]
        if key not in seen:
            seen.add(key)
            unique.append(m)
    return unique[:top_k]


def get_course_context(query: str) -> str:
    kb = load_course_knowledge_base()
    matches = find_matching_knowledge(query, kb)
    if not matches:
        return ""
    context = f"【课程上下文】来自《{kb['course']['name']}》\n"
    matched_chapters = set()
    for m in matches:
        ch_title = m["chapter"]
        if ch_title not in matched_chapters:
            matched_chapters.add(ch_title)
            for ch in kb["chapters"]:
                if ch["title"] == ch_title:
                    context += f"\n📖 {ch_title}（{ch.get('difficulty', '中级')}）：{'; '.join(ch.get('learning_objectives', []))}\n"
                    break
    context += "\n匹配知识点：\n"
    for m in matches[:5]:
        context += f"  - {m['knowledge_point']}（{m['difficulty']}）\n"
    return context


# ================== 1. 大模型调用函数（支持流式回调） ==================
def get_answer(messages, stream_callback: Optional[Callable[[str, bool], None]] = None):
    """
    调用讯飞星火大模型（流式）
    messages: [{"role": "user", "content": "你好"}, ...]
    stream_callback: 可选回调函数，接收 (content_chunk, is_first_content)
                    若为 None，则使用默认的 print 输出到控制台。
    返回完整的回复文本。
    """
    headers = {
        'Authorization': API_KEY,
        'content-type': "application/json"
    }
    body = {
        "model": "x1",
        "user": "user_id",
        "messages": messages,
        "stream": True,
        "tools": [
            {
                "type": "web_search",
                "web_search": {
                    "enable": True,
                    "search_mode": "deep"
                }
            }
        ]
    }
    full_response = ""
    is_first_content = True

    # 默认回调：控制台打印
    if stream_callback is None:
        def default_callback(chunk, first):
            if first:
                print("\n*******************以上为思维链内容，模型回复内容如下********************\n")
            print(chunk, end="")

        stream_callback = default_callback

    response = requests.post(URL, json=body, headers=headers, stream=True, timeout=120)
    response.raise_for_status()
    for chunk_line in response.iter_lines():
        if not chunk_line or chunk_line == b'[DONE]':
            continue
        line = chunk_line.decode('utf-8')
        if line.startswith('data:'):
            data_str = line[5:]
        else:
            data_str = line
        try:
            chunk = json.loads(data_str)
            delta = chunk['choices'][0]['delta']
            # 思维链内容（可选）
            if 'reasoning_content' in delta and delta['reasoning_content']:
                # 思维链部分不传递给回调，直接打印（或忽略）
                print(delta['reasoning_content'], end="")
            if 'content' in delta and delta['content']:
                content = delta['content']
                stream_callback(content, is_first_content)
                if is_first_content:
                    is_first_content = False
                full_response += content
        except Exception:
            pass
    print()  # 最终换行
    return full_response


# ================== 工具函数：鲁棒JSON提取 ==================
def extract_json(text: str) -> Optional[dict]:
    """从文本中稳健提取JSON对象，支持多种格式变化。"""
    if not text:
        return None
    # 尝试1：直接解析（如果模型输出纯JSON）
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    # 尝试2：用正则提取 {...} 块中最外层的大括号内容
    brace_depth = 0
    json_start = -1
    for i, ch in enumerate(text):
        if ch == '{':
            if brace_depth == 0:
                json_start = i
            brace_depth += 1
        elif ch == '}':
            brace_depth -= 1
            if brace_depth == 0 and json_start != -1:
                candidate = text[json_start:i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    # 可能是嵌套层数计算问题，继续尝试
                    pass
    # 尝试3：用正则匹配松散JSON（允许末尾逗号等常见问题）
    matches = list(re.finditer(r'\{[^{}]*\}', text, re.DOTALL))
    for m in reversed(matches):
        candidate = m.group()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            # 尝试修复常见JSON错误
            try:
                fixed = re.sub(r',\s*}', '}', candidate)  # 去掉末尾逗号
                fixed = re.sub(r',\s*]', ']', fixed)
                return json.loads(fixed)
            except json.JSONDecodeError:
                continue
    return None


# ================== 安全过滤 ==================
SENSITIVE_KEYWORDS = [
    '政治敏感', '色情', '暴力', '违法', '毒品', '枪支', '赌博',
]
# 简单的中文转义处理
_SENSITIVE_PATTERNS = [
    re.compile(re.escape(kw), re.IGNORECASE) for kw in SENSITIVE_KEYWORDS
]


def content_safety_check(text: str) -> Optional[str]:
    """检查输入是否包含敏感内容，返回 None 表示安全，返回提示信息表示命中。"""
    for pattern, kw in zip(_SENSITIVE_PATTERNS, SENSITIVE_KEYWORDS):
        if pattern.search(text):
            return f"抱歉，您输入的内容包含敏感词汇（{kw}），请重新提问。"
    return None


def output_safety_filter(text: str) -> str:
    """对模型输出进行安全检查，替换敏感内容。"""
    for pattern, kw in zip(_SENSITIVE_PATTERNS, SENSITIVE_KEYWORDS):
        if pattern.search(text):
            return "⚠️ 模型生成的内容包含不安全信息，已过滤。请重新尝试。"
    return text


# ================== 2. 定义状态（包含画像和学习规划） ==================
class StudentProfile(TypedDict):
    knowledge_base: str  # 初级/中级/高级
    learning_style: str  # 视觉型/听觉型/动手型
    weak_points: List[str]  # 薄弱知识点列表
    interest: str  # 兴趣方向
    learning_pace: str  # 快/中/慢
    interaction_summary: str  # 交互历史摘要


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    user_intent: str
    profile: Optional[StudentProfile]
    learning_plan: Optional[List[str]]
    course_context: Optional[str]
    resource_plan: Optional[str]


# ================== 3. 智能体节点（使用 get_answer 的回调输出） ==================

def profile_agent(state: AgentState) -> Dict[str, Any]:
    existing_profile = state.get("profile") or {}
    recent_msgs = state["messages"][-4:]
    context = "\n".join([f"{'用户' if isinstance(m, HumanMessage) else 'AI'}: {m.content}" for m in recent_msgs])

    # 安全过滤
    safety_hit = content_safety_check(context)
    if safety_hit:
        print(f"[安全过滤] {safety_hit}")
        return {"profile": existing_profile}

    prompt = f"""
你是一个教育画像分析专家。根据以下对话历史，提取学生的画像信息（JSON格式）。
已有画像（若无信息可忽略）：{json.dumps(existing_profile, ensure_ascii=False)}
对话历史：
{context}
输出必须是合法JSON，包含以下6个字段：
- "knowledge_base": 知识基础，取值为"初级"/"中级"/"高级"
- "learning_style": 学习风格，取值为"视觉型"/"听觉型"/"动手型"
- "weak_points": 薄弱知识点，字符串数组，例如 ["循环", "递归"]
- "interest": 兴趣方向，字符串
- "learning_pace": 学习节奏，取值为"快"/"中"/"慢"
- "interaction_summary": 交互历史摘要，简短一句话总结

只输出JSON对象，不要有任何其他解释或标记。
"""
    # 画像分析节点不需要流式输出，直接收集完整回复
    resp = get_answer([{"role": "user", "content": prompt}], stream_callback=None)
    new_profile = extract_json(resp)
    if new_profile is not None:
        merged = {**existing_profile, **new_profile}
        # 合并 weak_points 列表（新旧取并集）
        if "weak_points" in new_profile and isinstance(new_profile["weak_points"], list):
            existing_weak = existing_profile.get("weak_points", [])
            if isinstance(existing_weak, list):
                merged["weak_points"] = list(set(existing_weak + new_profile["weak_points"]))
        print(f"[画像更新成功] 当前画像: {json.dumps(merged, ensure_ascii=False)}")
    else:
        print(f"[画像解析失败] 保留旧画像. 原始返回前200字: {resp[:200]}")
        merged = existing_profile
    return {"profile": merged}


def classify_intent(state: AgentState) -> Dict[str, Any]:
    last_msg = state["messages"][-1].content
    prompt = f"""
判断以下用户输入的目的，只输出一个词语（greeting / resource / chat / tutor / evaluation）：
- greeting: 打招呼、问候、感谢、开场白
- resource: 请求生成学习资源，包含"生成资源"、"帮我练习"、"出题"、"讲解"、"代码案例"、"推荐学习材料"等关键词
- tutor: 请求辅导或解释，包含"为什么"、"我不懂"、"帮我解释"、"什么意思"、"怎么理解"、"讲解一下"、"辅导我"、"帮我分析"等疑问
- evaluation: 请求学习评估，包含"评估"、"测试我"、"考核"、"我的学习效果"、"检测"等关键词
- chat: 普通学术问题或闲聊
输入: {last_msg}
输出:
"""
    # 意图识别不需要流式
    resp = get_answer([{"role": "user", "content": prompt}], stream_callback=None).strip().lower()
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
    print(f"[意图识别] {intent}")
    return {"user_intent": intent}


def greeting_node(state: AgentState) -> Dict[str, Any]:
    user_msg = state["messages"][-1].content if state["messages"] else ""
    prompt = f"""你的身份是「AI智学」，一个智能学习助手，由多智能体系统驱动，提供个性化学习服务。
你的能力包括：生成学习资源（文档、导图、练习、案例、视频）、智能辅导、学习路径规划和效果评估。

用户说：{user_msg}

请根据用户输入自然回应：
- 如果用户问好（你好/嗨/早上好），热情问候；
- 如果用户问你是谁/你叫什么，请自我介绍（名称、身份、能做什么）；
- 否则正常回答。
回复简短自然（不超过80字）。"""
    print("\nAI: ", end="")

    def cb(chunk, first):
        if first:
            print("\n*******************以上为思维链内容，模型回复内容如下********************\n")
        print(chunk, end="")

    answer = get_answer([{"role": "user", "content": prompt}], stream_callback=cb)
    print()
    return {"messages": [AIMessage(content=answer)]}


def chat_node(state: AgentState) -> Dict[str, Any]:
    history = []
    profile = state.get("profile") or {}
    course_ctx = state.get("course_context", "")
    last_question = ""
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage):
            history.append({"role": "user", "content": msg.content})
            last_question = msg.content
        elif isinstance(msg, AIMessage):
            history.append({"role": "assistant", "content": msg.content})

    style_guide = {
        "视觉型": "多使用图表、代码高亮、结构化的Markdown格式",
        "听觉型": "多用类比和叙述性解释",
        "动手型": "鼓励动手实践，提供可运行的代码示例",
    }.get(profile.get("learning_style", "视觉型"), "")

    # 检查用户是否在询问身份
    identity_keywords = ["你是谁", "你叫什么", "你是什么", "你的名字", "你是哪个", "你的身份"]
    is_asking_identity = any(kw in last_question for kw in identity_keywords)

    system_prompt = f"""你是一个智能学习助手「AI智学」，提供个性化学习服务。
请根据以下学生画像调整回答方式：
- 知识基础：{profile.get('knowledge_base', '初级')}
- 学习风格：{profile.get('learning_style', '视觉型')}，{style_guide}
- 薄弱点：{', '.join(profile.get('weak_points', ['无']))}
- 兴趣方向：{profile.get('interest', '编程')}

{course_ctx}

请用适合学生水平和风格的方式回答问题。如果涉及学生薄弱点，请重点解释。"""

    # 如果用户在问身份，额外注入身份声明
    if is_asking_identity:
        identity_msg = {"role": "user", "content": "【系统指令】你的名字是「AI智学」，是一个智能学习助手。请直接告诉用户你的名字「AI智学」和你提供的功能。"}
        messages_with_system = [{"role": "system", "content": system_prompt}, identity_msg] + history
    else:
        messages_with_system = [{"role": "system", "content": system_prompt}] + history

    print("AI: ", end="")

    def console_callback(chunk, first):
        if first:
            print("\n*******************以上为思维链内容，模型回复内容如下********************\n")
        print(chunk, end="")

    answer = get_answer(messages_with_system, stream_callback=console_callback)
    answer = output_safety_filter(answer)
    plan = state.get("learning_plan")
    if plan:
        answer += "\n\n📌 **根据你的学习情况，推荐学习路径：**\n" + "\n".join(
            f"{i + 1}. {step}" for i, step in enumerate(plan))
    return {"messages": [AIMessage(content=answer)]}


def is_tutor_request(text: str) -> bool:
    """判断是否为辅导请求。"""
    keywords = ["为什么", "我不懂", "帮我解释", "什么意思", "怎么理解", "讲解一下", "辅导我", "帮我分析"]
    return any(kw in text for kw in keywords)


def tutor_node(state: AgentState) -> Dict[str, Any]:
    """智能辅导节点：当学生对某个知识点不理解时触发。"""
    last_question = state["messages"][-1].content
    profile = state.get("profile") or {}
    history = []
    for msg in state["messages"][-6:]:
        if isinstance(msg, HumanMessage):
            history.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            history.append({"role": "assistant", "content": msg.content})

    prompt = f"""
你是一个有耐心的智能辅导老师。学生提出了一个疑问，请从以下三个方面进行解答：

1. **核心概念解释**：用通俗易懂的语言解释相关知识点，结合学生画像调整讲解方式。
2. **图解/类比说明**：用ASCII图、流程图、表格或生活类比来辅助理解。
3. **学习建议**：如果学生仍有疑问，推荐下一步学习方向。

学生画像：
- 知识基础：{profile.get('knowledge_base', '初级')}
- 学习风格：{profile.get('learning_style', '视觉型')}
- 薄弱点：{', '.join(profile.get('weak_points', ['无']))}

学生提问：{last_question}

请在回复前标注 **[智能辅导]** 以表明这是辅导回复。
"""
    print("\n🧑‍🏫 [智能辅导节点] 正在为你详细解答...\n")

    def tutor_callback(chunk, first):
        if first:
            print("\n*******************以上为思维链内容，模型回复内容如下********************\n")
        print(chunk, end="")

    answer = get_answer([{"role": "user", "content": prompt}], stream_callback=tutor_callback)
    answer = output_safety_filter(answer)
    return {"messages": [AIMessage(content=answer)]}


def evaluation_node(state: AgentState) -> Dict[str, Any]:
    """学习效果评估节点：分析对话历史中的练习情况，生成评估报告。"""
    profile = state.get("profile") or {}
    history_text = ""
    for msg in state["messages"][-10:]:
        role = "用户" if isinstance(msg, HumanMessage) else "AI"
        content_preview = msg.content[:200]
        history_text += f"{role}: {content_preview}\n"

    prompt = f"""
你是一个学习效果评估专家。根据以下对话历史和画像，生成一份评估报告。

学生画像：
- 知识基础：{profile.get('knowledge_base', '初级')}
- 薄弱点：{', '.join(profile.get('weak_points', ['无']))}
- 学习风格：{profile.get('learning_style', '视觉型')}

最近对话历史（含练习记录）：
{history_text}

请输出评估报告，包含以下部分：
1. **知识掌握度**：评估学生对知识点的掌握程度（百分比估算）
2. **薄弱环节**：指出仍然存在的薄弱知识点
3. **进步情况**：与之前相比的进步
4. **学习建议**：下一步重点学习方向和建议
5. **推荐调整**：是否需要调整学习节奏或方法

在回复前标注 **[学习效果评估]**，要求内容具体、有针对性。
"""
    print("\n📊 [学习效果评估节点] 正在生成评估报告...\n")

    def eval_callback(chunk, first):
        if first:
            print("\n*******************以上为思维链内容，模型回复内容如下********************\n")
        print(chunk, end="")

    answer = get_answer([{"role": "user", "content": prompt}], stream_callback=eval_callback)
    answer = output_safety_filter(answer)
    return {"messages": [AIMessage(content=answer)]}


# ================== 资源生成多智能体协作组 ==================

def resource_planner_agent(state: AgentState) -> Dict[str, Any]:
    """资源规划智能体：分析画像和课程知识，规划资源内容。"""
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

    resp = get_answer([{"role": "user", "content": prompt}], stream_callback=None)
    plan = extract_json(resp)
    if plan:
        print(f"[资源规划] 主题：{plan.get('topic', '未指定')} | 难度：{plan.get('difficulty_level', '中级')}")
        return {"resource_plan": json.dumps(plan, ensure_ascii=False), "_resource_msg_index": len(state["messages"])}
    else:
        print("[资源规划] 使用默认规划")
        return {"resource_plan": resp, "_resource_msg_index": len(state["messages"])}


def content_generator_agent(state: AgentState) -> Dict[str, Any]:
    """内容生成智能体：生成讲解文档和练习题。"""
    profile = state.get("profile") or {}
    last_question = state["messages"][-1].content
    course_ctx = state.get("course_context", "")
    resource_plan = state.get("resource_plan", "")
    print("\n📝 [内容生成智能体] 正在生成讲解文档和练习题...\n")

    prompt = f"""你是一个教学内容生成专家。请生成讲解文档和练习题两种资源。

资源规划参考：{resource_plan}
{course_ctx}
学生画像：
- 知识基础：{profile.get('knowledge_base', '初级')}
- 学习风格：{profile.get('learning_style', '视觉型')}
- 薄弱点：{', '.join(profile.get('weak_points', ['无']))}
- 兴趣方向：{profile.get('interest', '编程')}

学生提问：{last_question}

请严格按照以下格式输出：

## 📘 1. 讲解文档
（详细文档：概念解释、核心原理、代码示例、复杂度分析、应用场景。针对薄弱点重点讲解。）

## 📝 2. 练习题
（至少3道：选择题2道+编程题1-2道，附答案和解析）"""

    def cb(chunk, first):
        if first:
            print("\n*******************以上为思维链内容，模型回复内容如下********************\n")
        print(chunk, end="")

    answer = get_answer([{"role": "user", "content": prompt}], stream_callback=cb)
    answer = output_safety_filter(answer)
    return {"messages": [AIMessage(content=answer)]}


def multimodal_generator_agent(state: AgentState) -> Dict[str, Any]:
    """多模态设计智能体：生成思维导图和视频脚本。"""
    profile = state.get("profile") or {}
    last_question = state["messages"][-1].content
    course_ctx = state.get("course_context", "")
    resource_plan = state.get("resource_plan", "")
    print("\n🎨 [多模态设计智能体] 正在生成思维导图和视频脚本...\n")

    prompt = f"""你是一个多模态教学内容设计专家。请生成思维导图和视频脚本。

资源规划参考：{resource_plan}
{course_ctx}
学生画像：
- 知识基础：{profile.get('knowledge_base', '初级')}
- 学习风格：{profile.get('learning_style', '视觉型')}
- 薄弱点：{', '.join(profile.get('weak_points', ['无']))}

学生提问：{last_question}

请严格按照以下格式输出：

## 🗺️ 3. 知识点思维导图
（文本描述，至少3个一级分支，每分支至少2个子节点，格式如下：
- 中心主题
  - 分支1：子节点1、子节点2...）

## 🎥 5. 多模态教学视频/动画脚本
（3-5个分镜场景，每个场景含：画面描述、旁白文本、动画效果建议、时长）"""

    def cb(chunk, first):
        if first:
            print("\n*******************以上为思维链内容，模型回复内容如下********************\n")
        print(chunk, end="")

    answer = get_answer([{"role": "user", "content": prompt}], stream_callback=cb)
    answer = output_safety_filter(answer)
    return {"messages": [AIMessage(content=answer)]}


def case_generator_agent(state: AgentState) -> Dict[str, Any]:
    """案例生成智能体：生成实操案例。"""
    profile = state.get("profile") or {}
    last_question = state["messages"][-1].content
    course_ctx = state.get("course_context", "")
    resource_plan = state.get("resource_plan", "")
    print("\n💻 [案例生成智能体] 正在生成实操案例...\n")

    prompt = f"""你是一个实践教学案例设计专家。请生成实操案例。

资源规划参考：{resource_plan}
{course_ctx}
学生画像：
- 知识基础：{profile.get('knowledge_base', '初级')}
- 学习风格：{profile.get('learning_style', '视觉型')}
- 薄弱点：{', '.join(profile.get('weak_points', ['无']))}
- 兴趣方向：{profile.get('interest', '编程')}

学生提问：{last_question}

请严格按照以下格式输出：

## 💻 4. 实操案例
（完整案例：案例名称、问题描述、需求分析、完整代码（带注释）、运行结果、扩展思考）"""

    def cb(chunk, first):
        if first:
            print("\n*******************以上为思维链内容，模型回复内容如下********************\n")
        print(chunk, end="")

    answer = get_answer([{"role": "user", "content": prompt}], stream_callback=cb)
    answer = output_safety_filter(answer)
    return {"messages": [AIMessage(content=answer)]}


def resource_merger_agent(state: AgentState) -> Dict[str, Any]:
    """资源合并智能体：合并各智能体输出，检查完整性。"""
    resource_parts = []
    # 仅扫描本轮资源生成中新添加的消息
    msg_start = state.get("_resource_msg_index", 0)
    for msg in reversed(state["messages"][msg_start:]):
        if isinstance(msg, AIMessage):
            content = msg.content
            if any(h in content for h in ["讲解文档", "知识点思维导图", "练习题", "实操案例", "多模态教学视频"]):
                resource_parts.append(content)
                if len(resource_parts) >= 4:
                    break
    combined = "\n\n".join(reversed(resource_parts))
    print("\n✅ [资源合并] 正在整合各智能体生成的资源...")

    resource_headers = ["讲解文档", "知识点思维导图", "练习题", "实操案例", "多模态教学视频"]
    missing = [h for h in resource_headers if h not in combined]
    if missing:
        print(f"[资源校验] 缺少: {missing}，补充生成...")
        supplement = get_answer([{"role": "user", "content": f"请补充以下资源：{missing}"}], stream_callback=None)
        combined += "\n\n--- 补充 ---\n" + supplement

    return {"messages": [AIMessage(content=combined)]}


def plan_node(state: AgentState) -> Dict[str, Any]:
    profile = state.get("profile") or {}
    course_ctx = state.get("course_context", "")
    weak_points = profile.get('weak_points', [])
    weak_text = ', '.join(weak_points) if weak_points else '暂无明确薄弱点'
    prompt = f"""
你是一个学习路径规划专家。根据学生画像和课程知识，规划个性化学习路径（3-5个步骤）。

画像信息：
- 知识基础：{profile.get('knowledge_base', '初级')}
- 学习风格：{profile.get('learning_style', '视觉型')}
- 薄弱点：{weak_text}
- 兴趣方向：{profile.get('interest', '编程')}
- 学习节奏：{profile.get('learning_pace', '中')}

{course_ctx}

要求：
1. 从课程知识体系中提取相关章节参考
2. 每个步骤包含**具体学习活动**和**预期目标**
3. **重点针对薄弱点**弥补
4. 根据学习风格推荐适合的资源类型
5. 遵循认知规律：先基础后深入、先理论后实践

输出格式（每行以数字序号开头）：
1. 第一步：[活动] - [描述] | 目标：[目标] | 资源：[类型]
2. 第二步：[活动] - [描述] | 目标：[目标] | 资源：[类型]
...
"""
    plan_text = get_answer([{"role": "user", "content": prompt}], stream_callback=None)
    steps = []
    for line in plan_text.split('\n'):
        line = line.strip()
        if line and len(line) > 2 and line[0].isdigit() and '.' in line:
            step_content = line.split('.', 1)[1].strip()
            if step_content:
                steps.append(step_content)
    if not steps:
        steps = [
            f"夯实基础 - 复习相关概念 | 目标：建立框架 | 资源：讲解文档",
            f"针对性练习 - 针对{weak_text}专项练习 | 目标：攻克薄弱 | 资源：练习题",
            f"综合实践 - 完成项目案例 | 目标：综合运用 | 资源：实操案例",
            f"巩固提升 - 梳理知识体系 | 目标：查漏补缺 | 资源：思维导图+视频"
        ]
    print(f"🗺️ [路径规划] 已生成 {len(steps)} 个学习步骤")
    return {"learning_plan": steps}


def route_after_classification(state: AgentState) -> str:
    intent = state["user_intent"]
    if intent == "greeting":
        return "greeting_node"
    elif intent == "resource":
        return "resource_planner_agent"
    else:
        # 在 chat 中判断是否为辅导或评估请求
        last_msg = state["messages"][-1].content if state["messages"] else ""
        if "评估" in last_msg and ("学习" in last_msg or "效果" in last_msg):
            return "evaluation_node"
        elif is_tutor_request(last_msg):
            return "tutor_node"
        return "chat_node"


# ================== 4. 构建图 ==================
def build_graph():
    workflow = StateGraph(AgentState)
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
            "resource_planner_agent": "resource_planner_agent"
        }
    )
    # 资源生成多智能体协作链路
    workflow.add_edge("resource_planner_agent", "content_generator_agent")
    workflow.add_edge("content_generator_agent", "multimodal_generator_agent")
    workflow.add_edge("multimodal_generator_agent", "case_generator_agent")
    workflow.add_edge("case_generator_agent", "resource_merger_agent")
    workflow.add_edge("resource_merger_agent", "plan_node")
    workflow.add_edge("plan_node", END)
    workflow.add_edge("greeting_node", END)
    workflow.add_edge("chat_node", END)
    workflow.add_edge("tutor_node", END)
    workflow.add_edge("evaluation_node", END)

    return workflow.compile()


# ================== 5. 交互运行 ==================
def run_interactive(app):
    print("🚀 个性化学习多智能体系统已启动（5种资源生成 + 学习路径规划 + 智能辅导 + 效果评估）")
    print("支持功能：")
    print("  - 输入任意内容开始对话，系统会自动分析画像")
    print("  - 输入'生成资源'或'帮我练习'触发5种资源生成+路径规划")
    print("  - 输入'评估我的学习效果'触发学习效果评估")
    print("  - 输入'为什么...'或'我不懂...'触发智能辅导")
    print("  - 输入 'exit' 退出程序\n")
    state = {"messages": [], "user_intent": "", "profile": None, "learning_plan": None, "course_context": "", "resource_plan": ""}
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