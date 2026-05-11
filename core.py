#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多智能体学习系统 — 公共核心模块
提供：配置加载、课程知识库、工具函数、状态定义、LLM 调用封装、安全过滤
"""

import os
import json
import re
import logging
import requests
from typing import TypedDict, Annotated, Dict, Any, List, Optional
from dotenv import load_dotenv
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

# ================== 日志配置 ==================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ai_learning")

# ================== 配置加载 ==================

load_dotenv()

API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not API_KEY:
    raise ValueError("请在 .env 文件中设置 DEEPSEEK_API_KEY")

API_URL = "https://api.deepseek.com/v1/chat/completions"
API_TIMEOUT = 120
API_MODEL = "deepseek-chat"

# ================== 课程知识库 ==================

_COURSE_KB: Optional[dict] = None


def load_course_knowledge_base() -> dict:
    """加载《数据结构与算法》课程知识库，带缓存。"""
    global _COURSE_KB
    if _COURSE_KB is not None:
        return _COURSE_KB
    kb_path = os.path.join(os.path.dirname(__file__), "course_knowledge_base.json")
    if os.path.exists(kb_path):
        with open(kb_path, "r", encoding="utf-8") as f:
            _COURSE_KB = json.load(f)
        logger.info("课程知识库已加载: %s (%s 章)", _COURSE_KB["course"]["name"], len(_COURSE_KB["chapters"]))
    else:
        logger.warning("课程知识库文件未找到，使用默认配置")
        _COURSE_KB = {"course": {"name": "默认课程"}, "chapters": [], "knowledge_graph": {}}
    return _COURSE_KB


def find_matching_knowledge(query: str, kb: dict, top_k: int = 3) -> List[dict]:
    """基于关键词匹配查询课程知识库。"""
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
                    "score": score,
                })
    matches.sort(key=lambda x: x["score"], reverse=True)
    seen: set = set()
    unique = []
    for m in matches:
        key = m["knowledge_point"]
        if key not in seen:
            seen.add(key)
            unique.append(m)
    return unique[:top_k]


def get_course_context(query: str) -> str:
    """根据用户查询获取相关课程上下文文本。"""
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


# ================== JSON 提取工具 ==================

def extract_json(text: str) -> Optional[dict]:
    """从模型输出中鲁棒提取 JSON 对象，支持 3 级容错策略。"""
    if not text:
        return None
    # 策略1：直接解析纯 JSON
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    # 策略2：嵌套括号追踪提取
    brace_depth = 0
    json_start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if brace_depth == 0:
                json_start = i
            brace_depth += 1
        elif ch == "}":
            brace_depth -= 1
            if brace_depth == 0 and json_start != -1:
                candidate = text[json_start : i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    pass
    # 策略3：正则松散匹配 + 容错修复
    for m in re.finditer(r"\{[^{}]*\}", text, re.DOTALL):
        candidate = m.group()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            try:
                fixed = re.sub(r",\s*\}", "}", candidate)
                fixed = re.sub(r",\s*\]", "]", fixed)
                return json.loads(fixed)
            except json.JSONDecodeError:
                continue
    return None


# ================== 安全过滤 ==================

SENSITIVE_KEYWORDS = [
    "政治敏感", "色情", "暴力", "违法", "毒品", "枪支", "赌博",
]

_SENSITIVE_PATTERNS = [
    re.compile(re.escape(kw), re.IGNORECASE) for kw in SENSITIVE_KEYWORDS
]

SENSITIVE_REPLACEMENT = "⚠️ 模型生成的内容包含不安全信息，已过滤。请重新尝试。"


def content_safety_check(text: str) -> Optional[str]:
    """检查输入是否包含敏感内容，返回警告信息或 None。"""
    for pattern, kw in zip(_SENSITIVE_PATTERNS, SENSITIVE_KEYWORDS):
        if pattern.search(text):
            logger.warning("安全过滤命中（输入）: %s", kw)
            return f"抱歉，您输入的内容包含敏感词汇（{kw}），请重新提问。"
    return None


def output_safety_filter(text: str) -> str:
    """对模型输出进行安全过滤，替换敏感内容。"""
    for pattern, kw in zip(_SENSITIVE_PATTERNS, SENSITIVE_KEYWORDS):
        if pattern.search(text):
            logger.warning("安全过滤命中（输出）: %s", kw)
            return SENSITIVE_REPLACEMENT
    return text


# ================== 状态定义 ==================

class StudentProfile(TypedDict):
    knowledge_base: str       # 初级/中级/高级
    learning_style: str       # 视觉型/听觉型/动手型
    weak_points: List[str]    # 薄弱知识点列表
    interest: str             # 兴趣方向
    learning_pace: str        # 快/中/慢
    interaction_summary: str  # 交互历史摘要


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    user_intent: str
    profile: Optional[StudentProfile]
    learning_plan: Optional[List[Any]]
    course_context: Optional[str]
    resource_plan: Optional[str]


# ================== 意图识别辅助 ==================

IDENTITY_KEYWORDS = ["你是谁", "你叫什么", "你是什么", "你的名字", "你是哪个", "你的身份"]

TUTOR_KEYWORDS = ["为什么", "我不懂", "帮我解释", "什么意思", "怎么理解", "讲解一下", "辅导我", "帮我分析"]


def is_tutor_request(text: str) -> bool:
    """判断用户输入是否为辅导请求。"""
    return any(kw in text for kw in TUTOR_KEYWORDS)


def is_identity_question(text: str) -> bool:
    """判断用户是否在询问助手的身份。"""
    return any(kw in text for kw in IDENTITY_KEYWORDS)


# ================== 学习风格映射 ==================

STYLE_GUIDE = {
    "视觉型": "多使用图表、代码高亮、结构化的Markdown格式",
    "听觉型": "多用类比和叙述性解释",
    "动手型": "鼓励动手实践，提供可运行的代码示例",
}


def get_style_guide(profile: Optional[dict]) -> str:
    """根据学生画像获取学习风格指导文本。"""
    if not profile:
        return STYLE_GUIDE.get("视觉型", "")
    return STYLE_GUIDE.get(profile.get("learning_style", "视觉型"), "")


# ================== LLM 调用封装 ==================

def _build_api_body(messages: List[dict]) -> dict:
    """构建 DeepSeek API 请求体（OpenAI 兼容格式）。"""
    return {
        "model": API_MODEL,
        "messages": messages,
        "stream": True,
    }


def _build_headers() -> dict:
    return {
        "Authorization": API_KEY,
        "content-type": "application/json",
    }


def call_llm_stream(messages: List[dict]):
    """
    流式调用 DeepSeek 大模型（生成器版），实时 yield token。

    yield 格式:
        ("chunk", content_text)   — 模型输出文本片段（仅 content 字段）
        ("status", status_text)   — 状态提示
        ("_llm_done", full_text)  — 调用完成，附带完整回复文本
        ("error", error_text)     — 出错时
    """
    full = ""
    try:
        resp = requests.post(API_URL, json=_build_api_body(messages), headers=_build_headers(),
                             stream=True, timeout=API_TIMEOUT)
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            s = line.decode("utf-8")
            if s.startswith("data: "):
                s = s[6:]
            if s == "[DONE]":
                continue
            try:
                d = json.loads(s)
                delta = d["choices"][0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    full += content
                    yield ("chunk", content)
            except (json.JSONDecodeError, KeyError, IndexError):
                pass
    except requests.Timeout:
        logger.error("LLM API 请求超时 (%ss)", API_TIMEOUT)
        yield ("error", "请求超时，请稍后重试")
    except requests.RequestException as e:
        logger.error("LLM API 请求失败: %s", e)
        yield ("error", f"服务暂时不可用：{e}")
    yield ("_llm_done", full)


def call_llm_sync(messages: List[dict]) -> str:
    """非流式调用 DeepSeek 大模型，返回完整回复文本。"""
    full = ""
    try:
        resp = requests.post(API_URL, json=_build_api_body(messages), headers=_build_headers(),
                             stream=True, timeout=API_TIMEOUT)
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            s = line.decode("utf-8")
            if s.startswith("data: "):
                s = s[6:]
            if s == "[DONE]":
                continue
            try:
                d = json.loads(s)
                content = d["choices"][0].get("delta", {}).get("content", "")
                full += content
            except (json.JSONDecodeError, KeyError, IndexError):
                pass
    except requests.Timeout:
        logger.error("LLM API 同步请求超时")
    except requests.RequestException as e:
        logger.error("LLM API 同步请求失败: %s", e)
    return full


def call_llm_console(messages: List[dict]):
    """
    控制台版流式调用 DeepSeek。实时打印回复内容。
    返回完整回复文本。
    """
    full_response = ""
    started = False

    headers = _build_headers()
    body = _build_api_body(messages)

    try:
        response = requests.post(API_URL, json=body, headers=headers, stream=True, timeout=API_TIMEOUT)
        response.raise_for_status()
        for chunk_line in response.iter_lines():
            if not chunk_line:
                continue
            s = chunk_line.decode("utf-8")
            if s.startswith("data: "):
                s = s[6:]
            if s == "[DONE]":
                continue
            try:
                d = json.loads(s)
                content = d["choices"][0].get("delta", {}).get("content", "")
                if content:
                    if not started:
                        started = True
                    print(content, end="", flush=True)
                    full_response += content
            except (json.JSONDecodeError, KeyError, IndexError):
                pass
        print()
    except requests.Timeout:
        logger.error("控制台版 API 请求超时")
        print("\n[错误] 请求超时，请稍后重试")
    except requests.RequestException as e:
        logger.error("控制台版 API 请求失败: %s", e)
        print(f"\n[错误] 服务暂时不可用：{e}")
    return full_response


# ================== B站视频搜索 ==================

def search_bilibili_videos(keyword: str, count: int = 3) -> List[Dict[str, Any]]:
    """
    调用 B站搜索 API，返回相关视频列表。

    返回字段: bvid, title, author, play, pic
    异常时安全降级返回空列表。
    """
    try:
        url = "https://api.bilibili.com/x/web-interface/search/type"
        params = {
            "search_type": "video",
            "keyword": keyword,
            "page": 1,
        }
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
            "Referer": "https://www.bilibili.com/",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        }
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != 0:
            logger.warning("B站 API 返回异常: code=%s, message=%s", data.get("code"), data.get("message", ""))
            return []
        result = data.get("data", {}).get("result", [])
        if not result:
            return []
        videos = []
        for item in result[:count]:
            videos.append({
                "bvid": item.get("bvid", ""),
                "title": item.get("title", "").replace("<em class=\"keyword\">", "").replace("</em>", ""),
                "author": item.get("author", ""),
                "play": item.get("play", 0),
                "pic": item.get("pic", ""),
            })
        logger.info("B站搜索 '%s' 返回 %d 个视频", keyword, len(videos))
        return videos
    except requests.Timeout:
        logger.warning("B站 API 请求超时")
    except requests.RequestException as e:
        logger.warning("B站 API 请求失败: %s", e)
    except Exception as e:
        logger.warning("B站搜索异常: %s", e)
    return []


# ================== 资源类型常量 ==================

RESOURCE_HEADERS = ["讲解文档", "知识点思维导图", "练习题", "实操案例", "多模态教学视频"]

# ================== Prompt 构建工具 ==================

def build_profile_prompt(existing_profile: dict, recent_context: str) -> str:
    """构建画像分析 prompt。"""
    return f"""你是一个教育画像分析专家。根据以下对话历史，提取学生的画像信息（JSON格式）。
已有画像（若无信息可忽略）：{json.dumps(existing_profile, ensure_ascii=False)}
对话历史：
{recent_context}
输出必须是合法JSON，包含以下6个字段：
- "knowledge_base": 知识基础，取值为"初级"/"中级"/"高级"
- "learning_style": 学习风格，取值为"视觉型"/"听觉型"/"动手型"
- "weak_points": 薄弱知识点，字符串数组，例如 ["循环", "递归"]
- "interest": 兴趣方向，字符串
- "learning_pace": 学习节奏，取值为"快"/"中"/"慢"
- "interaction_summary": 交互历史摘要，简短一句话总结

只输出JSON对象，不要有任何其他解释或标记。"""


def build_intent_prompt(last_message: str) -> str:
    """构建意图识别 prompt。"""
    return f"""判断以下用户输入的目的，只输出一个词语（greeting / resource / chat / tutor / evaluation）：
- greeting: 打招呼、问候、感谢、开场白
- resource: 请求生成学习资源，包含"生成资源"、"帮我练习"、"出题"、"讲解"、"代码案例"、"推荐学习材料"等关键词
- tutor: 请求辅导或解释，包含"为什么"、"我不懂"、"帮我解释"、"什么意思"、"怎么理解"、"讲解一下"、"辅导我"、"帮我分析"等疑问
- evaluation: 请求学习评估，包含"评估"、"测试我"、"考核"、"我的学习效果"、"检测"等关键词
- chat: 普通学术问题或闲聊
输入: {last_message}
输出:"""


def build_chat_system_prompt(profile: Optional[dict], course_ctx: str) -> str:
    """构建普通对话的系统 prompt。"""
    style_guide = get_style_guide(profile)
    return f"""你是一个智能学习助手「AI智学」，提供个性化学习服务。
请根据以下学生画像调整回答方式：
- 知识基础：{profile.get('knowledge_base', '初级') if profile else '初级'}
- 学习风格：{profile.get('learning_style', '视觉型') if profile else '视觉型'}，{style_guide}
- 薄弱点：{', '.join(profile.get('weak_points', ['无'])) if profile else '无'}
- 兴趣方向：{profile.get('interest', '编程') if profile else '编程'}

{course_ctx}

请用适合学生水平和风格的方式回答问题。如果涉及学生薄弱点，请重点解释。"""


def build_greeting_prompt(user_msg: str) -> str:
    """构建问候 prompt，包含身份声明。"""
    return f"""你的身份是「AI智学」，一个智能学习助手，由多智能体系统驱动，提供个性化学习服务。
你的能力包括：生成学习资源（文档、导图、练习、案例、视频）、智能辅导、学习路径规划和效果评估。

用户说：{user_msg}

请根据用户输入自然回应：
- 如果用户问好（你好/嗨/早上好），热情问候；
- 如果用户问你是谁/你叫什么，请自我介绍（名称、身份、能做什么）；
- 否则正常回答。
回复简短自然（不超过80字）。"""


def build_tutor_prompt(last_question: str, profile: Optional[dict], course_ctx: str) -> str:
    """构建智能辅导 prompt。"""
    return f"""你是一个有耐心的智能辅导老师。学生提出了一个疑问，请从以下三个方面进行解答：

1. **核心概念解释**：用通俗易懂的语言解释相关知识点，结合学生画像调整讲解方式。
2. **图解/类比说明**：用ASCII图、流程图、表格或生活类比来辅助理解。
3. **学习建议**：如果学生仍有疑问，推荐下一步学习方向。

学生画像：
- 知识基础：{profile.get('knowledge_base', '初级') if profile else '初级'}
- 学习风格：{profile.get('learning_style', '视觉型') if profile else '视觉型'}
- 薄弱点：{', '.join(profile.get('weak_points', ['无'])) if profile else '无'}

{course_ctx}

学生提问：{last_question}

请在回复前标注 **[智能辅导]** 以表明这是辅导回复。"""


def build_evaluation_prompt(profile: Optional[dict], history_text: str) -> str:
    """构建学习效果评估 prompt（要求 JSON 输出）。"""
    return f"""你是一个学习效果评估专家。根据以下对话历史和画像，生成一份结构化的学习效果评估。

学生画像：
- 知识基础：{profile.get('knowledge_base', '初级') if profile else '初级'}
- 薄弱点：{', '.join(profile.get('weak_points', ['无'])) if profile else '无'}
- 学习风格：{profile.get('learning_style', '视觉型') if profile else '视觉型'}

最近对话历史（含练习记录）：
{history_text}

请输出 JSON 对象，包含以下字段：
1. "overall_score": 整数 0-100，综合评估分数
2. "knowledge_level": 知识掌握度百分比，如 "75%"
3. "efficiency_level": 学习效率描述，如 "较高"、"中等"、"偏低"
4. "weak_points_list": 薄弱知识点数组，如 ["变量作用域", "递归"]
5. "progress_summary": 进步情况简述
6. "suggestions": 学习建议
7. "pace_recommendation": 节奏调整建议

只输出JSON对象，不要有任何其他解释或标记。"""


def build_plan_prompt(profile: Optional[dict], course_ctx: str) -> str:
    """构建学习路径规划 prompt。"""
    weak_points = profile.get("weak_points", []) if profile else []
    weak_text = ", ".join(weak_points) if weak_points else "暂无明确薄弱点"
    return f"""你是一个学习路径规划专家。根据学生画像和课程知识，规划个性化学习路径（3-5个步骤）。

画像信息：
- 知识基础：{profile.get('knowledge_base', '初级') if profile else '初级'}
- 学习风格：{profile.get('learning_style', '视觉型') if profile else '视觉型'}
- 薄弱点：{weak_text}
- 兴趣方向：{profile.get('interest', '编程') if profile else '编程'}
- 学习节奏：{profile.get('learning_pace', '中') if profile else '中'}

{course_ctx}

要求：
1. 从课程知识体系中提取相关章节参考
2. 每个步骤包含**具体学习活动**和**预期目标**
3. **重点针对薄弱点**弥补
4. 根据学习风格推荐适合的资源类型
5. 遵循认知规律：先基础后深入、先理论后实践

输出JSON对象，包含两个字段：
1. "summary": 路径总结（一句话）
2. "steps": 步骤数组，每个元素包含：
   - "title": 步骤标题（简短）
   - "description": 详细描述（包含具体活动）
   - "goal": 学习目标
   - "resource_types": 推荐资源类型数组

只输出JSON对象，不要有任何其他解释或标记。"""


# ================== 画像合并 ==================

def merge_profile(existing: dict, new_profile: dict) -> dict:
    """合并新旧画像，weak_points 取并集。"""
    merged = {**existing, **new_profile}
    if "weak_points" in new_profile and isinstance(new_profile["weak_points"], list):
        existing_weak = existing.get("weak_points", [])
        if isinstance(existing_weak, list):
            merged["weak_points"] = list(set(existing_weak + new_profile["weak_points"]))
    return merged
