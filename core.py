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
import subprocess
import requests
from typing import TypedDict, Annotated, Dict, Any, List, Optional
from dotenv import load_dotenv
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from web_search_client import search_open_websearch, get_websearch_client

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

BING_API_KEY = os.getenv("BING_API_KEY", "")
BING_SEARCH_URL = "https://api.bing.microsoft.com/v7.0/search"

# ================== 豆包(Seedream)图片生成 API ==================

DOUBAO_API_KEY = os.getenv("DOUBAO_API_KEY", "ark-2a059d2d-ce46-45f2-b3f2-8e2d69a9052f-1db26")
DOUBAO_API_URL = "https://ark.cn-beijing.volces.com/api/v3/images/generations"
DOUBAO_MODEL = "doubao-seedream-5-0-260128"

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


def get_user_documents_context(session_id: str, query: str | None = None) -> str:
    """加载用户上传的知识库文档，返回上下文文本，可选关键词匹配筛选。"""
    from database import get_documents_content
    try:
        docs = get_documents_content(session_id)
        if not docs:
            return ""

        # 如果有查询关键词，进行简单匹配筛选
        if query:
            scored = []
            ql = query.lower()
            for d in docs:
                score = 0
                if ql in d.get("title", "").lower():
                    score += 10
                if ql in d.get("content", "").lower():
                    score += 1
                if score > 0:
                    scored.append((score, d))
            scored.sort(key=lambda x: x[0], reverse=True)
            docs = [d for _, d in scored[:3]]

        if not docs:
            return ""

        parts = ["【用户知识库】"]
        for d in docs:
            parts.append(f"\n--- 文档: {d['title']} ---\n{d['content']}\n---")

        ctx = "\n".join(parts)
        max_len = 4000
        if len(ctx) > max_len:
            ctx = ctx[:max_len] + "\n\n... (内容过长，已截断)"
        return ctx
    except Exception as e:
        logger.warning("加载用户文档失败: %s", e)
        return ""


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
    knowledge_base: str        # 初级/中级/高级
    learning_style: str        # 视觉型/听觉型/动手型/读写型
    cognitive_style: str       # 分析型/直觉型/反思型/实践型
    weak_points: List[str]     # 薄弱知识点列表
    error_patterns: List[str]  # 易错点偏好/常见错误类型
    interest: str              # 兴趣方向/专业领域
    learning_pace: str         # 快/中/慢
    learning_goals: str        # 学习目标
    motivation_level: str      # 学习动机水平: 高/中/低
    interaction_summary: str   # 交互历史摘要


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
    "读写型": "提供详细的文字说明、文档结构和理论知识",
}


def get_style_guide(profile: Optional[dict]) -> str:
    """根据学生画像获取学习风格 + 认知风格指导文本。"""
    if not profile:
        return "多使用图表、代码高亮、结构化的Markdown格式"
    guide = STYLE_GUIDE.get(profile.get("learning_style", "视觉型"), "")
    # 叠加认知风格指导
    cog = profile.get("cognitive_style", "")
    if cog == "分析型":
        guide += "；注重逻辑推导和分步拆解"
    elif cog == "直觉型":
        guide += "；先给整体框架再深入细节"
    elif cog == "反思型":
        guide += "；多提问引导思考，留出反思空间"
    elif cog == "实践型":
        guide += "；以实际案例驱动，边做边学"
    return guide


# ================== LLM 调用封装 ==================

def _build_api_body(messages: List[dict], **extra) -> dict:
    """构建 DeepSeek API 请求体（OpenAI 兼容格式）。"""
    body = {
        "model": API_MODEL,
        "messages": messages,
        "stream": True,
    }
    body.update(extra)
    return body


def _build_headers() -> dict:
    return {
        "Authorization": API_KEY,
        "content-type": "application/json",
    }


def call_llm_stream(messages: List[dict], **kwargs):
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
        resp = requests.post(API_URL, json=_build_api_body(messages, **kwargs), headers=_build_headers(),
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


def call_llm_sync(messages: List[dict], **kwargs) -> str:
    """非流式调用 DeepSeek 大模型，返回完整回复文本。"""
    full = ""
    try:
        resp = requests.post(API_URL, json=_build_api_body(messages, **kwargs), headers=_build_headers(),
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


# ================== Bing 联网搜索 ==================

def search_bing(query: str, count: int = 5) -> List[Dict[str, str]]:
    """
    调用 Bing Web Search API，返回搜索结果列表。

    返回字段: title, url, snippet
    异常或无 API Key 时安全降级返回空列表。
    """
    if not BING_API_KEY:
        logger.warning("BING_API_KEY 未配置，跳过联网搜索")
        return []
    try:
        headers = {
            "Ocp-Apim-Subscription-Key": BING_API_KEY,
            "Accept-Language": "zh-CN,zh;q=0.9",
        }
        params = {
            "q": query,
            "count": count,
            "mkt": "zh-CN",
            "textFormat": "Raw",
        }
        resp = requests.get(BING_SEARCH_URL, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        web_pages = data.get("webPages", {}).get("value", [])
        if not web_pages:
            logger.info("Bing 搜索 '%s' 无结果", query)
            return []
        results = []
        for item in web_pages[:count]:
            results.append({
                "title": item.get("name", ""),
                "url": item.get("url", ""),
                "snippet": item.get("snippet", ""),
            })
        logger.info("Bing 搜索 '%s' 返回 %d 条结果", query, len(results))
        return results
    except requests.Timeout:
        logger.warning("Bing API 请求超时")
    except requests.RequestException as e:
        logger.warning("Bing API 请求失败: %s", e)
    except Exception as e:
        logger.warning("Bing 搜索异常: %s", e)
    return []


def search_duckduckgo(query: str, count: int = 5) -> List[Dict[str, str]]:
    """
    调用 DuckDuckGo 搜索。完全免费，无需 API Key。
    优先 ddgs 库，其次 Lite HTML 抓取。
    """
    # 策略1: ddgs / duckduckgo_search 库
    DDGS_cls = None
    for mod_name in ("ddgs", "duckduckgo_search"):
        try:
            DDGS_cls = __import__(mod_name, fromlist=["DDGS"]).DDGS
            break
        except (ImportError, AttributeError):
            continue
    if DDGS_cls is not None:
        try:
            results = []
            with DDGS_cls() as ddgs:
                for r in ddgs.text(query, max_results=count, safesearch="off"):
                    results.append({
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "snippet": r.get("body", ""),
                    })
            if results:
                logger.info("DDGS 库搜索 '%s' 返回 %d 条结果", query, len(results))
                return results
        except Exception as e:
            logger.warning("DDGS 库搜索异常: %s", e)

    # 策略2: DuckDuckGo Lite HTML 页面
    try:
        from bs4 import BeautifulSoup
        url = "https://lite.duckduckgo.com/lite/"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Content-Type": "application/x-www-form-urlencoded",
        }
        resp = requests.post(url, data={"q": query}, headers=headers, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        results = []
        for row in soup.select("table tbody tr")[:count * 2]:
            links = row.select("a.result-link")
            snippets = row.select("td.result-snippet")
            if links:
                title = links[0].get_text(strip=True)
                href = links[0].get("href", "")
                snippet = snippets[0].get_text(strip=True) if snippets else ""
                if title and href:
                    results.append({"title": title, "url": href, "snippet": snippet})
                    if len(results) >= count:
                        break
        if results:
            logger.info("DDG Lite '%s' 返回 %d 条结果", query, len(results))
            return results
    except ImportError:
        pass
    except Exception as e:
        logger.warning("DDG Lite 搜索异常: %s", e)
    return []


def _search_bing_web(query: str, count: int = 5) -> List[Dict[str, str]]:
    """抓取 Bing 网页搜索结果（无需 API Key，国内可访问）。"""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return []
    try:
        url = "https://www.bing.com/search"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
            "Accept-Language": "zh-CN,zh;q=0.9",
        }
        params = {"q": query, "count": count, "setlang": "zh-Hans"}
        resp = requests.get(url, headers=headers, params=params, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        results = []
        for item in soup.select("li.b_algo")[:count]:
            title_el = item.select_one("h2 a")
            snippet_el = item.select_one(".b_caption p")
            if title_el:
                results.append({
                    "title": title_el.get_text(strip=True),
                    "url": title_el.get("href", ""),
                    "snippet": snippet_el.get_text(strip=True) if snippet_el else "",
                })
        logger.info("Bing 网页抓取 '%s' 返回 %d 条结果", query, len(results))
        return results
    except Exception as e:
        logger.warning("Bing 网页抓取异常: %s", e)
    return []


# ================== 搜索查询优化 ==================

# 新闻/资讯类关键词 — 明确在查询新闻、热点、实时动态
# 注意："今天/昨天/前天/后天" 不在此列 —— 它们只是时间指代，不应触发新闻模式
_NEWS_KEYWORDS = [
    "新闻", "热点", "热搜", "头条", "资讯", "报道", "动态",
    "大事", "最新消息", "实时", "刚刚",
]

# 时间指代词 — 需要转换为具体日期以避免搜索引擎歧义（如"昨天"→泰剧）
# 但与新闻无关，不应触发信息摘要模式
_TIME_KEYWORDS = ["今天", "昨天", "前天", "后天", "明天",
                  "近日", "本周", "最近", "最近发生"]

# 搜索噪声词 — 从查询中移除（会干扰搜索引擎的动词/虚词）
_SEARCH_NOISE = ["有什么", "有哪些", "帮我查", "帮我搜", "帮我找", "帮我",
                  "我想知道", "我想了解", "请问", "请告诉我",
                  "搜索一下", "查找一下", "找一下", "查一下",
                  "搜索", "查找", "获取", "查询", "我想查",
                  "是什么", "什么是", "有没有", "能不能",
                  "告诉我", "的"]

# 纯日期模式
_DATE_PATTERN = r'\d{4}年\d{1,2}月\d{1,2}日'


def _is_news_query(query: str) -> bool:
    """判断用户是否在明确查询新闻/实时资讯（包含新闻/热点/头条等关键词）"""
    return any(kw in query for kw in _NEWS_KEYWORDS)


def _has_time_reference(query: str) -> bool:
    """判断查询是否包含时间指代词（今天/昨天/最近等）"""
    return any(kw in query for kw in _TIME_KEYWORDS)


def optimize_search_query(query: str, force_time_replace: bool = True) -> str:
    """
    优化搜索查询，提升搜索精度：
    - 时间词 → 具体日期（避免搜到同名影视作品）
    - 去除噪声词，提炼核心关键词
    - 新闻查询追加语境
    """
    import datetime
    import re
    optimized = query.strip()

    # 去除噪声词
    for noise in _SEARCH_NOISE:
        optimized = optimized.replace(noise, "")

    # 清理多余空格和标点残留
    optimized = re.sub(r'\s+', ' ', optimized).strip()
    optimized = optimized.strip("，。！？、的了吗呢")

    # 时间词 → 日期替换（无论是否新闻查询，都做，避免搜到同名影视作品）
    if force_time_replace or _has_time_reference(query):
        today = datetime.date.today()
        optimized = optimized.replace("前天",
                                       (today - datetime.timedelta(days=2)).strftime("%Y年%m月%d日"))
        optimized = optimized.replace("昨天",
                                       (today - datetime.timedelta(days=1)).strftime("%Y年%m月%d日"))
        optimized = optimized.replace("今天", today.strftime("%Y年%m月%d日"))
        optimized = optimized.replace("明天",
                                       (today + datetime.timedelta(days=1)).strftime("%Y年%m月%d日"))
        optimized = optimized.replace("后天",
                                       (today + datetime.timedelta(days=2)).strftime("%Y年%m月%d日"))

    # 明确新闻查询：追加语境
    if _is_news_query(query):
        if "新闻" not in optimized:
            optimized = f"{optimized} 新闻"

    # 纯日期查询（日期后没有具体主题词）→ 补充搜索语境
    _TOPIC_WORDS = ["新闻", "热点", "资讯", "动态", "科技", "财经", "体育", "娱乐", "要闻",
                    "节日", "纪念日", "日子", "是什么", "事件"]
    has_topic = any(tw in optimized for tw in _TOPIC_WORDS)
    if re.search(_DATE_PATTERN, optimized) and len(optimized.replace(" ", "")) < 15 and not has_topic:
        optimized = f"{optimized} 大事件 节日"

    logger.info("搜索查询优化: '%s' → '%s'", query, optimized)
    return optimized


def search_web(query: str, count: int = 5) -> List[Dict[str, str]]:
    """
    统一联网搜索入口。自动优化查询。
    降级链：open-webSearch(本地) → Bing API → Bing 网页抓取 → DuckDuckGo
    """
    # 优化查询词
    optimized = optimize_search_query(query)

    # 策略0: open-webSearch 本地服务（多引擎，免费，无需 API Key）
    results = search_open_websearch(optimized, count)
    if results:
        return results

    # 策略1: Bing API（有 Key 时最快最稳定）
    if BING_API_KEY:
        results = search_bing(optimized, count)
        if results:
            return results

    # 策略2: Bing 网页抓取（国内可访问，无需 Key）
    results = _search_bing_web(optimized, count)
    if results:
        return results

    # 策略3: DuckDuckGo（国外可用，免费免注册）
    return search_duckduckgo(optimized, count)


# ================== 网页内容富化 ==================

def _fetch_page_content_open_websearch(url: str, timeout: int = 15) -> str:
    """通过 open-webSearch MCP 服务抓取网页内容。速度快，但遇到反爬/JS页面会失败。"""
    try:
        client = get_websearch_client()
        result = client.fetch_web(url, max_chars=10000, readability=True, include_links=False)
        content = (
            result.get("content", "")
            or result.get("text", "")
            or result.get("raw", "")
        )
        if content and len(str(content).strip()) > 50:
            return str(content)[:10000]
        return ""
    except Exception as e:
        logger.warning("open-webSearch 抓取网页失败 (%s): %s", url[:60], e)
        return ""


def _fetch_page_content_browseract(url: str, timeout: int = 30) -> str:
    """通过 browser-act stealth-extract 抓取网页内容。反检测隐身，能处理JS渲染，
    但首次运行需启动Chrome，较慢（15-30秒）。"""
    try:
        result = subprocess.run(
            ["browser-act", "stealth-extract", url, "--format", "markdown"],
            capture_output=True, text=True, timeout=timeout, encoding="utf-8",
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
        else:
            stderr_snippet = result.stderr[:200] if result.stderr else "(无输出)"
            logger.warning("browser-act 抓取失败 (%s): %s", url[:60], stderr_snippet)
            return ""
    except subprocess.TimeoutExpired:
        logger.warning("browser-act 抓取超时 (%ds): %s", timeout, url[:60])
        return ""
    except FileNotFoundError:
        logger.warning("browser-act 命令不可用，请确认已安装")
        return ""
    except Exception as e:
        logger.warning("browser-act 抓取异常 (%s): %s", url[:60], e)
        return ""


def fetch_page_content(url: str, timeout: int = 30) -> str:
    """统一入口。降级链：open-webSearch → browser-act → 空字符串"""
    # 策略1: open-webSearch (本地MCP，最快)
    content = _fetch_page_content_open_websearch(url, min(timeout, 15))
    if content:
        return content

    # 策略2: browser-act stealth-extract (反检测，可处理JS页面)
    content = _fetch_page_content_browseract(url, timeout)
    if content:
        return content

    return ""


def enrich_search_results(results: List[Dict], top_n: int = 2) -> List[Dict]:
    """为 top N 条搜索结果抓取完整页面内容，存入 'content' 字段。
    静默跳过失败——不因页面抓取失败影响基础搜索。"""
    if not results:
        return results
    for i, result in enumerate(results[:top_n]):
        url = result.get("url", "")
        if not url:
            continue
        logger.info("📄 正在获取页面内容 (%d/%d): %s",
                     i + 1, min(top_n, len(results)), url[:80])
        content = fetch_page_content(url)
        if content:
            result["content"] = content
            logger.info("  ✓ 获取 %d 字", len(content))
        else:
            logger.info("  ✗ 跳过（无法获取内容）")
    return results


def format_search_context(results: List[Dict[str, str]],
                          user_query: str = "") -> str:
    """将搜索结果格式化为注入 LLM 的上下文字符串。
    三种模式：新闻摘要 / 搜索增强对话 / 无搜索"""
    if not results:
        return ""
    import datetime
    today = datetime.date.today().strftime("%Y年%m月%d日")

    is_news = _is_news_query(user_query)

    lines = [
        f"[联网搜索结果 | 今天是{today}]",
        "",
    ]

    if is_news:
        # 新闻/资讯模式：角色切换为「信息摘要助手」，禁止展开教学
        lines += [
            "⚠️ 你是一个**信息摘要助手**（不是教学专家）。用户正在查询实时资讯。",
            "你必须：",
            "  1. 简洁地列出搜索结果中的热点/新闻（每条1-2句话即可）",
            "  2. 引用来源标题和链接",
            "  3. **禁止生成教学文档、练习题、思维导图、视频脚本等教学材料**",
            "  4. 如果搜索结果不包含用户想要的资讯，直接告知，禁止编造",
            "  5. 用「以下是今日热点资讯汇总」的风格输出，不要用「讲解文档」风格",
            "  6. **重要**：用户查询中的'昨天''今天'等词永远指代时间，不是任何影视作品名称。"
            "禁止提及泰剧《昨天》、电影《昨天》或任何与搜索结果无关的文艺作品。",
            "",
        ]
    else:
        # 搜索增强模式：搜索结果作为辅助参考，非强制性
        # 适用于：学习类查询 + 通用知识查询（如"今天是什么日子"）
        lines += [
            "⚠️ 以下是从搜索引擎获取的实时信息，可作为辅助参考：",
            "  1. 优先参考搜索结果中的事实性信息（日期、数据、事件等）",
            "  2. 在回答中自然融入相关信息，必要时引用来源",
            "  3. 如果搜索结果是相关的，就如实引用；如果不够充分，可以结合你的知识补充",
            "  4. 禁止编造搜索结果中不存在的具体数字、日期、人名、事件名",
            "  5. 如有帮助可以引用来源链接",
            "",
        ]

    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r['title']}")
        lines.append(f"   {r['snippet']}")
        lines.append(f"   来源: {r['url']}")
        # 追加页面完整内容（富化后才有 'content' 字段）
        if r.get("content"):
            page_text = str(r["content"])[:3000]
            lines.append(f"   📄 页面内容: {page_text}")
        lines.append("")
    return "\n".join(lines)


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
    """构建画像分析 prompt — 10 维度深度抽取。"""
    return f"""你是一个教育画像分析专家。请根据以下对话历史，深度提取学生的学习画像信息。

## 已有画像（参考，可能过时）
{json.dumps(existing_profile, ensure_ascii=False, indent=2)}

## 对话历史
{recent_context}

## 分析要求
从对话中自动推断以下维度。若对话中无明确证据，根据上下文合理推断，不要留空：

输出必须是合法 JSON，包含以下 **10 个字段**：

1. "knowledge_base": 知识基础水平 → "入门"/"初级"/"中级"/"高级"/"精通"
2. "learning_style": 学习风格偏好 → "视觉型"/"听觉型"/"动手型"/"读写型"
   - 视觉型：喜欢图表、代码、可视化
   - 听觉型：喜欢讲解、讨论、叙述
   - 动手型：喜欢实操、编程练习、项目
   - 读写型：喜欢文档、笔记、理论
3. "cognitive_style": 认知风格 → "分析型"/"直觉型"/"反思型"/"实践型"
   - 分析型：喜欢拆解问题、逻辑推理
   - 直觉型：喜欢类比跳跃、快速把握全局
   - 反思型：喜欢反复思考、追问为什么
   - 实践型：喜欢直接上手、边做边学
4. "weak_points": 薄弱知识点列表，例如 ["递归", "动态规划", "指针"]
5. "error_patterns": 易错点偏好/常见错误模式列表，例如 ["边界条件遗漏", "时间复杂度误判", "语法细节疏忽"]
6. "interest": 兴趣方向/专业领域，例如 "后端开发"、"算法竞赛"、"AI/机器学习"
7. "learning_pace": 学习节奏 → "快"/"中"/"慢"
8. "learning_goals": 学习目标，例如 "备战秋招面试"、"通过期末考试"、"转行学编程"
9. "motivation_level": 学习动机水平 → "高"/"中"/"低"
10. "interaction_summary": 交互历史一句话摘要，概括学生的关注点和行为特征

## 重要规则
- 每个字段都必须有值，不允许 null 或空字符串
- 数组字段至少包含一个元素，若不够确定则填入合理推断值
- 结合已有画像的演变趋势，体现"随学随新"

只输出 JSON 对象，不要有任何其他解释或标记。"""


def build_intent_prompt(last_message: str) -> str:
    """构建意图识别 prompt。"""
    return f"""判断以下用户输入的目的，只输出一个词语（greeting / resource / chat / tutor / evaluation）：
- greeting: 打招呼、问候、感谢、开场白
- resource: 请求生成学习资源，包含"生成资源"、"帮我练习"、"出题"、"讲解"、"代码案例"、"推荐学习材料"等关键词
  注意：如果用户是在查询新闻、热点、资讯、实时动态（如"今天有什么新闻"、"昨天热点"），这不是resource，应判为chat。
- tutor: 请求辅导或解释，包含"为什么"、"我不懂"、"帮我解释"、"什么意思"、"怎么理解"、"讲解一下"、"辅导我"、"帮我分析"等疑问
- evaluation: 请求学习评估，包含"评估"、"测试我"、"考核"、"我的学习效果"、"检测"等关键词
- chat: 普通学术问题、闲聊、新闻资讯查询、实时信息查询
输入: {last_message}
输出:"""


def build_chat_system_prompt(profile: Optional[dict], course_ctx: str) -> str:
    """构建普通对话的系统 prompt。
    三模式：信息摘要（新闻查询）/ 搜索增强对话 / 纯学习助手"""
    style_guide = get_style_guide(profile)
    has_search = "[联网搜索结果" in course_ctx
    is_news_mode = "信息摘要助手" in course_ctx

    base_profile = (
        f"学生画像：\n"
        f"- 知识基础：{profile.get('knowledge_base', '初级') if profile else '初级'}\n"
        f"- 学习风格：{profile.get('learning_style', '视觉型') if profile else '视觉型'}，{style_guide}\n"
        f"- 薄弱点：{', '.join(profile.get('weak_points', ['无'])) if profile else '无'}\n"
        f"- 兴趣方向：{profile.get('interest', '编程') if profile else '编程'}"
    )

    # 模式1：新闻/资讯摘要
    if is_news_mode:
        return (
            f"你是一个智能信息摘要助手「AI智学」，帮助用户快速了解实时资讯。\n"
            f"根据以下学生画像调整语言难度：\n{base_profile}\n\n"
            f"{course_ctx}\n\n"
            f"请简洁、结构化地呈现信息。用条目式输出，每条新闻配一句话摘要+来源链接。\n"
            f"禁止生成教学文档、练习题、思维导图等教学材料。"
        )

    # 模式2：搜索增强对话（非新闻的搜索查询）
    if has_search:
        return (
            f"你是一个智能学习助手「AI智学」，提供个性化学习服务。\n"
            f"{base_profile}\n\n"
            f"{course_ctx}\n\n"
            f"【注意】上述 [联网搜索结果] 包含来自搜索引擎的实时信息。\n"
            f"请将这些信息作为辅助参考融入回答中，优先引用搜索结果中的事实数据。\n"
            f"如果搜索结果充分且相关，就以此为基础回答；如果不够充分，可结合你的知识补充。\n"
            f"坚持你的「学习助手」身份，不要变成搜索引擎的复读机。\n"
            f"如果涉及学生薄弱点，请重点解释。")
    # 模式3：纯学习助手（无搜索）
    return (
        f"你是一个智能学习助手「AI智学」，提供个性化学习服务。\n"
        f"{base_profile}\n\n"
        f"{course_ctx}\n\n"
        f"请用适合学生水平和风格的方式回答问题。如果涉及学生薄弱点，请重点解释。"
    )


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
    has_search = "[联网搜索结果" in course_ctx
    search_note = ""
    if has_search:
        search_note = (
            "\n⚠️ 上述 [联网搜索结果] 是搜索引擎返回的实时信息，回答时必须优先参考。禁止编造。\n"
        )
    return f"""你是一个有耐心的智能辅导老师。学生提出了一个疑问，请从以下三个方面进行解答：

1. **核心概念解释**：用通俗易懂的语言解释相关知识点，结合学生画像调整讲解方式。
2. **图解/类比说明**：用ASCII图、流程图、表格或生活类比来辅助理解。
3. **学习建议**：如果学生仍有疑问，推荐下一步学习方向。

学生画像：
- 知识基础：{profile.get('knowledge_base', '初级') if profile else '初级'}
- 学习风格：{profile.get('learning_style', '视觉型') if profile else '视觉型'}
- 薄弱点：{', '.join(profile.get('weak_points', ['无'])) if profile else '无'}

{course_ctx}
{search_note}
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


# ================== 单资源聚焦生成 Prompt ==================

_SINGLE_RESOURCE_PROMPTS = {
    "doc": (
        "你是一份教学讲解文档生成器。你的输出将直接被保存为DOCX文件，交付给学生阅读。\n\n"
        "⛔ 铁律——违反即为失败输出：\n"
        "1. 你的回复中不得出现以下任何词汇：选择题、编程题、练习题、习题、题目、测验、试题、问答、填空、判断\n"
        "2. 不得输出任何需要学生作答的内容（包括选项A/B/C/D、答案、解析）\n"
        "3. 不得输出 ```mermaid 代码块（思维导图）\n"
        "4. 不得出现\"场景\"+\"旁白\"+\"画面描述\"的组合（视频脚本）\n"
        "5. 禁止任何问候语、开场白、结尾总结语——直接从 ## 概念解释 开始写\n\n"
        "✅ 唯一允许的输出结构（直接从下面第一行开始，不要任何前缀）：\n"
        "## 概念解释\n（用通俗易懂的语言解释核心概念，配合生活类比）\n\n"
        "## 原理说明\n（深入讲解工作原理和机制，可配合ASCII图示）\n\n"
        "## 代码示例\n（完整可运行的代码，带关键注释，标注语言）\n\n"
        "## 复杂度分析\n（时间复杂度和空间复杂度分析，用表格呈现）\n\n"
        "## 应用场景\n（实际工程中的应用场景和最佳实践）"
    ),
    "exercise": (
        "你是一份练习题生成器。你的输出将直接被保存为DOCX文件，交付给学生练习。\n\n"
        "⛔ 铁律——违反即为失败输出：\n"
        "1. 禁止任何知识讲解、概念解释、原理说明——学生已经学过，不需要你再教一遍\n"
        "2. 你的回复中不得出现以下章节标题：概念解释、原理说明、知识回顾、学习目标、课前导读、背景知识\n"
        "3. 禁止任何问候语、开场白、鼓励语、结尾总结——直接输出题目\n"
        "4. 禁止输出 ```mermaid 代码块\n"
        "5. 禁止出现\"场景\"+\"旁白\"+\"画面描述\"的组合\n\n"
        "✅ 唯一允许的输出结构（直接从下面第一行开始，不要任何前缀）：\n"
        "## 选择题\n**1. 题目描述**\nA. 选项A  B. 选项B  C. 选项C  D. 选项D\n答案：X\n解析：简要说明\n\n（至少5道选择题，覆盖不同知识点）\n\n"
        "## 编程题\n**题目：**具体题目描述（含输入输出格式）\n**示例：**输入→输出\n**参考代码：**```python\n代码\n```\n**思路：**解题思路简述\n\n（至少2道编程题，难度递进）\n\n"
        "## 难度分级\n- 基础：题号X、Y\n- 进阶：题号Z\n- 挑战：题号W"
    ),
    "mindmap": (
        "你是一个知识图谱可视化专家。你的唯一任务是生成关于用户指定主题的 **Mermaid mindmap 思维导图**。\n\n"
        "⛔ 你必须严格遵守以下规则，违反任何一条就是失败：\n"
        "1. **只输出 mindmap** —— 整个回复就是一个 ```mermaid 代码块，代码块外没有任何文字\n"
        "2. **只使用 mindmap 语法** —— 严禁 graph/flowchart/sequenceDiagram/classDiagram 等其他语法\n"
        "3. **结构要求**：根节点用 ((双括号))，至少4个一级分支，每个分支至少2个子节点\n"
        "4. **节点文字是纯文本** —— 不要用 [ ] 或 ( ) 包裹节点文字\n"
        "5. 不要有任何问候语、解释、总结或任何非代码块的内容\n\n"
        "示例格式（不要照抄内容）：\n"
        "```mermaid\nmindmap\n  root((主题名称))\n    分支1\n      子节点A\n      子节点B\n    分支2\n      子节点C\n      子节点D\n    分支3\n      子节点E\n      子节点F\n    分支4\n      子节点G\n      子节点H\n```"
    ),
    "video": (
        "你是一个教学视频编导专家。你的唯一任务是生成关于用户指定主题的**教学视频脚本**。\n\n"
        "必须包含以下结构：\n"
        "## 场景1：开场引入\n- 画面描述：...\n- 旁白：...\n- 动画效果：...\n- 时长：约30秒\n\n"
        "## 场景2：核心讲解\n- 画面描述：...\n- 旁白：...\n- 动画效果：...\n- 时长：约90秒\n\n"
        "## 场景3：代码/实例演示\n- 画面描述：...\n- 旁白：...\n- 动画效果：...\n- 时长：约60秒\n\n"
        "## 场景4：总结回顾\n- 画面描述：...\n- 旁白：...\n- 动画效果：...\n- 时长：约30秒\n\n"
        "⛔ 硬性规则：\n"
        "1. 只生成视频脚本，不生成讲解文档、不生成练习题、不生成思维导图\n"
        "2. 使用 Markdown 格式，标题从 ## 开始\n"
        "3. 不要输出任何与主题无关的内容"
    ),
    "case": (
        "你是一个实战案例教学专家。你的唯一任务是生成关于用户指定主题的**实操案例**。\n\n"
        "必须包含以下结构：\n"
        "## 案例名称\n（一个具体的、贴近实际工作的案例标题）\n\n"
        "## 问题描述\n（详细的业务场景和问题说明）\n\n"
        "## 需求分析\n（功能需求、技术需求、约束条件）\n\n"
        "## 完整代码\n（可直接运行的完整代码，包含详细注释）\n\n"
        "## 运行结果\n（展示运行输出或效果）\n\n"
        "## 扩展思考\n（2-3个扩展方向或优化建议）\n\n"
        "⛔ 硬性规则：\n"
        "1. 只生成实操案例，不生成讲解文档、不生成练习题、不生成思维导图\n"
        "2. 使用 Markdown 格式，标题从 ## 开始\n"
        "3. 不要输出任何与主题无关的内容"
    ),
    "ppt": (
        "你是一个教学PPT设计专家。你的唯一任务是生成关于用户指定主题的**PPT幻灯片大纲**。\n\n"
        "必须包含以下结构（每页用 ## 标记）：\n"
        "## 封面：{主题}\n- 副标题\n- 演讲者信息留白\n\n"
        "## 目录\n- 本章要点1\n- 本章要点2\n- 本章要点3\n- 本章要点4\n\n"
        "（以下为内容页，至少6页，每页3-5个要点）\n"
        "## 概念引入\n- 要点1\n- 要点2\n- 要点3\n\n"
        "## 核心原理\n- 要点1\n- 要点2\n- 要点3\n- 要点4\n\n"
        "## 代码示例\n- 要点1\n- 要点2\n- 要点3\n\n"
        "## 应用场景\n- 要点1\n- 要点2\n- 要点3\n\n"
        "## 常见误区\n- 要点1\n- 要点2\n- 要点3\n\n"
        "## 总结\n- 核心收获1\n- 核心收获2\n- 核心收获3\n\n"
        "⛔ 硬性规则：\n"
        "1. 只生成PPT大纲，用 ## 标记每页标题，用 - 标记要点\n"
        "2. 不生成讲解文档、不生成练习题、不生成思维导图\n"
        "3. 不要输出任何与主题无关的内容"
    ),
}


def build_single_resource_prompt(topic: str, resource_type: str) -> str:
    """构建单资源聚焦生成的系统提示词。返回 (system_prompt, user_prompt)。"""
    system_prompt = _SINGLE_RESOURCE_PROMPTS.get(
        resource_type,
        _SINGLE_RESOURCE_PROMPTS["doc"]
    )
    user_prompt = f"主题：{topic}"
    return system_prompt, user_prompt


def build_plan_prompt(profile: Optional[dict], course_ctx: str) -> str:
    """构建学习路径规划 prompt — 10 维度深度分析 + 科学规划。

    综合分析：画像基础(知识/风格/认知/动机/兴趣/薄弱点/易错点/节奏/目标)
    → 诊断学习现状 → 规划动态步骤 → 指定顺序与资源。
    """
    if not profile:
        profile = {}

    def _get(key, default="未知"):
        val = profile.get(key, "")
        if isinstance(val, list):
            return ", ".join(val) if val else "暂无"
        return val if val else default

    return f"""你是一位资深学习路径规划专家。请基于以下多维度学生画像，进行深度分析并规划个性化学习路径。

## 学生画像（10 维度）

| 维度 | 描述 |
|------|------|
| 知识基础 | {_get('knowledge_base')} |
| 学习风格 | {_get('learning_style')} |
| 认知风格 | {_get('cognitive_style')} |
| 学习节奏 | {_get('learning_pace')} |
| 学习动机 | {_get('motivation_level')} |
| 兴趣方向 | {_get('interest')} |
| 学习目标 | {_get('learning_goals')} |
| 薄弱点 | {_get('weak_points')} |
| 易错点 | {_get('error_patterns')} |
| 交互摘要 | {_get('interaction_summary')} |

## 课程上下文
{course_ctx if course_ctx else "（暂无课程上下文，请根据画像和通用学习规律规划）"}

## 分析要求

### 第一步：诊断分析
根据画像深度分析学生的学习现状：
- 当前处于什么学习阶段？（入门/进阶/冲刺）
- 最需要优先解决的知识缺口是什么？
- 认知风格决定了什么样的学习顺序最优？
- 动机水平对应什么样的难度梯度？

### 第二步：路径规划（4-6 个步骤）
根据诊断结果，规划科学的学习路径：
1. **难度递进**：从易到难，每个步骤建立在前一步基础上
2. **针对性**：优先针对薄弱点和易错点设计专项训练
3. **风格适配**：
   - 分析型 → 强调逻辑推导、分步拆解
   - 直觉型 → 先给全局框架再深入细节
   - 反思型 → 多设思考题、对比分析
   - 实践型 → 以项目和案例驱动
4. **动机匹配**：低动机→趣味案例入门；高动机→高强度挑战
5. **节奏对齐**：快节奏→紧凑安排；慢节奏→充分留白巩固
6. **目标导向**：明确每步与学习目标的对应关系

### 第三步：步骤设计
每个步骤应包含：
- 具体可执行的学习活动（不是泛泛的概念）
- 对应的资源类型推荐
- 明确的完成标准（什么情况下算掌握）
- 预估时间投入

## 输出格式

输出 JSON 对象，包含：
1. "diagnosis": 诊断分析，一段话总结学生当前学习状态
2. "summary": 路径总结（一句话）
3. "steps": 步骤数组（4-6个），每个包含：
   - "title": 步骤标题（简短，< 12 字）
   - "description": 详细描述（含具体活动和完成标准，80-150 字）
   - "goal": 学习目标
   - "resource_types": 推荐资源类型数组（如 ["讲解文档","练习题","实操案例","思维导图","教学视频"]）
   - "estimated_time": 预估学习时间（如 "2-3天" / "1周" / "4-6小时"）

只输出 JSON 对象，不要有任何其他解释或标记。"""


# ================== PPT 生成 ==================

def build_ppt_prompt(markdown_content: str, title: str) -> str:
    """构建 PPT 内容生成 prompt。"""
    return f"""你是教学PPT设计专家。请将以下学习资料转换为8-12页PPT的JSON结构。

PPT标题：{title}

学习资料内容：
{markdown_content[:8000]}

输出JSON对象，包含以下字段：
1. "title": PPT标题
2. "slides": 幻灯片数组，每页包含：
   - "title": 页面标题（必填）
   - "bullets": 要点列表（2-4条，每条简洁明了）
   - "notes": 讲师备注（可选，补充讲解要点）
   - "layout": 布局类型，取值为 "title"（封面）/ "content"（内容）/ "two_column"（双栏）/ "summary"（总结）

幻灯片结构要求：
- 第1页：封面（标题 + 副标题 + "AI智学出品"）
- 第2页：目录/大纲
- 第3页起：核心内容（每页聚焦一个知识点，2-4个要点）
- 倒数第2页：总结回顾
- 最后1页：Q&A + 致谢

每个要点文字控制在20字以内，简洁有力。
只输出JSON对象，不要有任何其他解释或标记。"""


def parse_ppt_json(raw: str) -> Optional[dict]:
    """从 LLM 输出中提取 PPT JSON，降级处理。"""
    data = extract_json(raw)
    if not data:
        return None
    if "slides" not in data:
        # 尝试修复：整个数据就是一个 slides 数组
        if isinstance(data, list):
            return {"title": "学习资源", "slides": data}
        return None
    return data


# ================== 文本安全截断 ==================

def _safe_truncate(text: str, max_len: int) -> str:
    """安全截断文本：闭合括号、词边界截断、移除尾部残缺标点。"""
    if len(text) <= max_len:
        return text
    result = text[:max_len]
    # 闭合未配对的括号
    for left, right in [("（", "）"), ("(", ")"), ("【", "】"), ("[", "]"), ("《", "》"), ("<", ">")]:
        if result.count(left) > result.count(right):
            # 找到最后一个左括号位置并截断到它之前
            last_left = result.rfind(left)
            if last_left >= 0 and (max_len - last_left) < 8:
                result = result[:last_left].rstrip()
                break
            else:
                result += right
    # 移除末尾不完整的标点/字符
    result = result.rstrip("，,。.、；;：:（(（[【《<")
    # 如果以英文单词截断，退回到上一个空格
    if result and result[-1].isascii() and result[-1].isalpha():
        last_space = result.rfind(" ")
        if last_space > max_len // 2:
            result = result[:last_space].rstrip()
    return result


def _wrap_text_line(text: str, font, max_width: int, draw) -> str:
    """按像素宽度自动换行，超长部分用 … 截断。"""
    if not text:
        return ""
    lines = []
    current = ""
    for ch in text:
        test = current + ch
        bbox = draw.textbbox((0, 0), test, font=font)
        w = bbox[2] - bbox[0]
        if w > max_width and current:
            lines.append(current)
            current = ch
        else:
            current = test
    if current:
        lines.append(current)
    # 限制行数，最后一行加 …
    if len(lines) > 3:
        lines = lines[:3]
        if lines[-1]:
            lines[-1] = _safe_truncate(lines[-1], len(lines[-1]) - 1) + "…"
    return "\n".join(lines)


# ================== 视频生成 ==================

def build_video_prompt(question: str, reference: str) -> str:
    """构建视频脚本 prompt——以用户问题为核心，参考资料仅供准确性参考。"""
    return f"""编写一份教学视频脚本，回答以下问题：

用户提问：
{question}

参考资料（仅用于确保准确性，不可直接复制其中的句子）：
{reference[:6000] if reference else "（无参考资料，请依据知识作答）"}

生成 5-8 个场景的 JSON 数组。每个场景必须严格对应：

1. 封面场景（scene=1）：
   - title: 从用户提问中提取核心关键词（≤12字），如提问是"Python怎么入门"，则 title="Python入门"
   - slide_text: 视频标题（≤18字），提炼用户问题的核心
   - narration: 一句话说明本视频将解答什么问题

2. 核心讲解场景（scene=2~N-2，3~5个）：
   - title: 知识点名称（≤10字）
   - slide_text: 该知识点的关键词/定义/公式（2-3行，每行≤15字），必须与 narration 内容一致
   - narration: 对该知识点的完整讲解（2-4句话），内容必须覆盖 slide_text 中的关键词

3. 总结场景（倒数第2个）：
   - slide_text: 核心结论（1行）
   - narration: 归纳各知识点间的关系，给出整体性的理解

4. 结尾场景（最后1个）：
   - slide_text: 建议的下一步行动
   - narration: 结束语（1-2句）

语体要求：
- 严肃、准确、专业。禁止口语语气词。
- 禁止：以"好的""收到""首先""接着""接下来"开头
- 禁止："你""大家""同学们""朋友们"
- narration 可用"本视频""本节""学习者"或直接陈述

格式：[{{"scene":1,"title":"...","slide_text":"...","narration":"..."}}]

只输出 JSON 数组。"""


def parse_video_script(raw: str) -> list:
    """从 LLM 输出中提取视频脚本 JSON 数组。"""
    data = extract_json(raw)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        # 可能是 {{"scenes": [...]}}
        scenes = data.get("scenes") or data.get("slides")
        if isinstance(scenes, list):
            return scenes
    return []


def sanitize_video_scenes(scenes: list, title: str) -> list:
    """清洗视频场景：修复封面标题异常（AI 幻觉文本替换为实际标题）。"""
    if not scenes:
        return scenes

    # AI 幻觉检测关键词
    HALLUCINATION_SIGNALS = [
        "收到", "指令", "作为", "专属", "拒绝", "好的",
        "根据您", "让我", "为您", "您的问题", "正在",
    ]
    first = scenes[0]
    scene_title = first.get("title", "")
    scene_slide = first.get("slide_text", "")

    # 检测封面标题是否异常
    is_bad = False
    if len(scene_title) > 20:
        is_bad = True
    else:
        for signal in HALLUCINATION_SIGNALS:
            if signal in scene_title:
                is_bad = True
                break

    if is_bad:
        clean_title = _safe_truncate(title, 12)
        first["title"] = clean_title
        # 也修复 slide_text，如果过长或含幻觉信号
        if len(scene_slide) > 30:
            first["slide_text"] = clean_title
        elif any(s in scene_slide for s in HALLUCINATION_SIGNALS):
            first["slide_text"] = clean_title

    return scenes


def render_video_frames(scenes: list, title: str, output_dir: str) -> list:
    """用 Pillow 将每个场景渲染为 PNG 图片帧（1920x1080）。"""
    from PIL import Image, ImageDraw, ImageFont
    import os as _os

    _os.makedirs(output_dir, exist_ok=True)
    W, H = 1920, 1080

    # 配色
    BG_DARK = (15, 23, 42)
    BG_LIGHT = (248, 250, 252)
    PRIMARY = (37, 99, 235)
    ACCENT = (236, 72, 153)
    WHITE = (255, 255, 255)
    DARK_TEXT = (30, 41, 59)
    MUTED = (148, 163, 184)

    frames = []

    # 尝试加载中文字体
    font_paths = [
        "C:\\Windows\\Fonts\\msyh.ttc",
        "C:\\Windows\\Fonts\\simhei.ttf",
        "C:\\Windows\\Fonts\\simsun.ttc",
    ]
    font_title = None
    font_body = None
    font_small = None
    for fp in font_paths:
        if _os.path.exists(fp):
            try:
                font_title = ImageFont.truetype(fp, 64)
                font_body = ImageFont.truetype(fp, 36)
                font_small = ImageFont.truetype(fp, 24)
                break
            except Exception:
                continue
    if font_title is None:
        font_title = ImageFont.load_default()
        font_body = font_title
        font_small = font_title

    for idx, scene in enumerate(scenes):
        is_title = (idx == 0)
        is_end = (idx == len(scenes) - 1)

        img = Image.new("RGB", (W, H), BG_DARK if (is_title or is_end) else BG_LIGHT)
        draw = ImageDraw.Draw(img)

        scene_title = scene.get("title", f"场景 {idx + 1}")
        slide_text = scene.get("slide_text", "")
        if isinstance(slide_text, list):
            slide_text = "\n".join(slide_text)

        if is_title:
            # 封面
            draw.rectangle([0, 0, W, H], fill=BG_DARK)
            draw.ellipse([W - 500, -100, W + 200, 700], fill=PRIMARY)
            draw.ellipse([100, H - 350, 450, H + 50], fill=ACCENT)
            # 标题 — 长标题自动换行
            title_wrapped = _wrap_text_line(
                _safe_truncate(title, 30), font_title, W - 240, draw)
            y_title = 260
            for tline in title_wrapped.split("\n"):
                draw.text((120, y_title), tline, fill=WHITE, font=font_title)
                y_title += 72
            draw.rectangle([120, y_title + 5, 380, y_title + 13], fill=ACCENT)
            draw.text((120, y_title + 30), scene_title, fill=MUTED, font=font_body)
            draw.text((120, 700), "AI智学 · 多智能体学习平台", fill=MUTED, font=font_small)
            draw.text((120, 750), f"共 {len(scenes)} 个场景 · 约 {len(scenes) * 40} 秒", fill=MUTED, font=font_small)

        elif is_end:
            # 结尾页
            draw.rectangle([0, 0, W, H], fill=BG_DARK)
            draw.ellipse([150, H - 350, 500, H + 50], fill=PRIMARY)
            draw.text((120, 300), "总结回顾", fill=WHITE, font=font_title)
            draw.rectangle([120, 440, 280, 448], fill=ACCENT)
            y = 520
            for line in slide_text.split("\n")[:5]:
                if line.strip():
                    draw.ellipse([120, y + 8, 136, y + 24], fill=ACCENT)
                    wrapped = _wrap_text_line(line.strip(), font_body, W - 300, draw)
                    for wline in wrapped.split("\n")[:2]:
                        draw.text((150, y), wline, fill=MUTED, font=font_body)
                        y += 42
                    y += 8
            draw.text((120, 750), "感谢观看 · AI智学出品", fill=MUTED, font=font_small)

        else:
            # 内容页
            draw.rectangle([0, 0, W, H], fill=BG_LIGHT)
            # 顶部导航条
            draw.rectangle([0, 0, W, 4], fill=PRIMARY)
            # 左侧色条
            draw.rectangle([0, 0, 8, H], fill=PRIMARY)
            # 标题背景
            draw.rectangle([80, 40, W - 80, 140], fill=WHITE)
            draw.rounded_rectangle([80, 40, W - 80, 140], radius=16, fill=WHITE,
                                   outline=(226, 232, 240), width=1)
            # 标题 — 按宽度自动换行
            wrap_title = _wrap_text_line(scene_title, ImageFont.truetype(font_title.path, 48)
                          if hasattr(font_title, 'path') else font_body, W - 700, draw)
            y_t = 60
            for tline in wrap_title.split("\n"):
                draw.text((130, y_t), tline, fill=DARK_TEXT,
                          font=ImageFont.truetype(font_title.path, 48)
                          if hasattr(font_title, 'path') else font_body)
                y_t += 50
            draw.rectangle([130, y_t + 5, 320, y_t + 9], fill=ACCENT)

            # 内容卡片
            card_y = max(y_t + 30, 180)
            card_h = min(700, 120 + max(1, slide_text.count("\n")) * 80)
            draw.rounded_rectangle([80, card_y, W - 500, card_y + card_h], radius=16,
                                   fill=WHITE, outline=(226, 232, 240), width=1)
            # 步骤编号
            draw.ellipse([80, card_y - 30, 140, card_y + 30], fill=PRIMARY)
            step_num = str(idx + 1)
            bbox = font_body.getbbox(step_num) if hasattr(font_body, 'getbbox') else (0, 0, 30, 30)
            tw = bbox[2] - bbox[0] if hasattr(font_body, 'getbbox') else 30
            draw.text((110 - tw // 2, card_y - 22), step_num, fill=WHITE, font=font_body)

            # 要点 — 按像素宽度自动换行
            card_text_width = W - 600
            y_text = card_y + 30
            for line in slide_text.split("\n")[:5]:
                if line.strip():
                    wrapped = _wrap_text_line(line.strip(), font_body, card_text_width, draw)
                    for wline in wrapped.split("\n")[:2]:
                        if wline.strip():
                            draw.ellipse([130, y_text + 12, 146, y_text + 28], fill=PRIMARY)
                            draw.text((160, y_text + 5), wline, fill=DARK_TEXT, font=font_body)
                            y_text += 50
                    y_text += 10

            # 右侧信息卡
            draw.rounded_rectangle([W - 400, card_y, W - 80, card_y + 200], radius=16,
                                   fill=PRIMARY)
            draw.text((W - 350, card_y + 30), f"第 {idx + 1} / {len(scenes)} 页",
                       fill=(191, 219, 254), font=font_small)
            draw.text((W - 350, card_y + 100), "AI智学",
                       fill=(191, 219, 254), font=font_body)
            draw.text((W - 350, card_y + 150), "多智能体学习平台",
                       fill=(191, 219, 254), font=font_small)

            # 底部进度条
            progress_w = int((W - 160) * (idx + 1) / len(scenes))
            draw.rectangle([80, H - 60, W - 80, H - 50], fill=(226, 232, 240))
            draw.rectangle([80, H - 60, 80 + progress_w, H - 50], fill=PRIMARY)

        filepath = _os.path.join(output_dir, f"frame_{idx:03d}.png")
        img.save(filepath, "PNG")
        frames.append(filepath)

    return frames


async def generate_video_narration(scenes: list, output_dir: str, voice: str = "zh-CN-XiaoxiaoNeural") -> list:
    """用 Edge-TTS 为每个场景生成旁白 MP3，返回音频文件路径列表。"""
    import edge_tts
    import os as _os

    _os.makedirs(output_dir, exist_ok=True)
    audio_files = []

    for idx, scene in enumerate(scenes):
        narration = scene.get("narration", "")
        if not narration:
            audio_files.append(None)
            continue
        filepath = _os.path.join(output_dir, f"narration_{idx:03d}.mp3")
        try:
            communicate = edge_tts.Communicate(narration, voice)
            await communicate.save(filepath)
            audio_files.append(filepath)
        except Exception as e:
            logger.warning("Edge-TTS 生成失败 (场景 %d): %s", idx, e)
            audio_files.append(None)

    return audio_files


def compose_video(frames: list, audio_files: list, output_path: str,
                  default_duration: float = 5.0) -> bool:
    """用 ffmpeg 将图片帧和旁白 MP3 合成 MP4。"""
    import subprocess
    import shutil
    import os as _os

    # 查找 ffmpeg
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        for base in ["C:\\ffmpeg\\bin", _os.path.expandvars(r"%LOCALAPPDATA%\\JianyingPro"),
                      _os.path.expandvars(r"%PROGRAMFILES%\\ffmpeg\\bin"),
                      _os.path.expandvars(r"%LOCALAPPDATA%\\Microsoft\\WinGet\\Packages")]:
            if "WinGet" in base:
                # 搜索 WinGet 下的 ffmpeg 安装
                try:
                    for pkg in _os.listdir(base):
                        if pkg.lower().startswith("gyan.ffmpeg"):
                            for root2, _, files2 in _os.walk(_os.path.join(base, pkg)):
                                for f2 in files2:
                                    if f2.lower() == "ffmpeg.exe":
                                        ffmpeg = _os.path.join(root2, f2)
                                        break
                                if ffmpeg:
                                    break
                    if ffmpeg:
                        break
                except Exception:
                    pass
                continue
            for root, _, files in _os.walk(base) if _os.path.isdir(base) else []:
                for f in files:
                    if f.lower() == "ffmpeg.exe":
                        ffmpeg = _os.path.join(root, f)
                        break
    if not ffmpeg:
        logger.error("未找到 ffmpeg")
        return False

    # 获取每帧持续时长（从 MP3 文件）
    import subprocess as sp
    def _get_duration(filepath):
        if not filepath or not _os.path.exists(filepath):
            return default_duration
        try:
            result = sp.run(
                [ffmpeg, "-i", filepath],
                capture_output=True, text=True, encoding="utf-8", errors="replace"
            )
            for line in (result.stderr or "").split("\n"):
                if "Duration" in line:
                    parts = line.split("Duration: ")[1].split(",")[0].split(":")
                    return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        except Exception:
            pass
        return default_duration

    durations = [_get_duration(af) for af in audio_files]

    # 构建 concat 文件（每帧 + 空音频拼接）
    concat_lines = []
    for i, (frame, audio, dur) in enumerate(zip(frames, audio_files, durations)):
        concat_lines.append(f"file '{_os.path.abspath(frame).replace(chr(92), '/')}'")
        concat_lines.append(f"duration {dur:.3f}")

    concat_file = _os.path.join(_os.path.dirname(output_path), "concat.txt")
    with open(concat_file, "w", encoding="utf-8") as f:
        f.write("\n".join(concat_lines))

    # 合成视频流（静态帧 → 视频）
    video_raw = output_path.replace(".mp4", "_video.mp4")
    cmd_video = [
        ffmpeg, "-y",
        "-f", "concat", "-safe", "0", "-i", concat_file,
        "-vsync", "cfr", "-pix_fmt", "yuv420p",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-r", "1", video_raw
    ]
    try:
        sp.run(cmd_video, check=True, capture_output=True, timeout=120, encoding="utf-8", errors="replace")
    except sp.CalledProcessError as e:
        logger.error("ffmpeg 视频合成失败: %s", e.stderr[:200] if e.stderr else str(e))
        if _os.path.exists(concat_file):
            _os.unlink(concat_file)
        return False

    # 合成完整音频流
    audio_concat_file = _os.path.join(_os.path.dirname(output_path), "audio_concat.txt")
    with open(audio_concat_file, "w", encoding="utf-8") as f:
        for af in audio_files:
            if af and _os.path.exists(af):
                f.write(f"file '{_os.path.abspath(af).replace(chr(92), '/')}'\n")

    audio_raw = output_path.replace(".mp4", "_audio.mp3")
    # 如果有有效音频文件，则合并
    valid_audio = [af for af in audio_files if af and _os.path.exists(af)]
    if valid_audio:
        cmd_audio = [
            ffmpeg, "-y",
            "-f", "concat", "-safe", "0", "-i", audio_concat_file,
            "-c:a", "libmp3lame", "-q:a", "2", audio_raw
        ]
        try:
            sp.run(cmd_audio, check=True, capture_output=True, timeout=60, encoding="utf-8", errors="replace")
        except sp.CalledProcessError:
            logger.warning("音频合并失败，使用静音")
            valid_audio = []

    # 合并视频 + 音频
    if valid_audio:
        cmd_merge = [
            ffmpeg, "-y",
            "-i", video_raw, "-i", audio_raw,
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-shortest", "-movflags", "+faststart",
            output_path
        ]
    else:
        cmd_merge = [
            ffmpeg, "-y",
            "-i", video_raw,
            "-c:v", "copy",
            "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
            "-c:a", "aac", "-b:a", "64k",
            "-shortest", "-movflags", "+faststart",
            output_path
        ]

    try:
        sp.run(cmd_merge, check=True, capture_output=True, timeout=60, encoding="utf-8", errors="replace")
    except sp.CalledProcessError as e:
        logger.error("ffmpeg 合并失败: %s", e.stderr[:200] if e.stderr else str(e))
        return False

    # 清理临时文件
    for tmp in [concat_file, audio_concat_file, video_raw, audio_raw]:
        if _os.path.exists(tmp):
            try:
                _os.unlink(tmp)
            except Exception:
                pass

    return _os.path.exists(output_path)


def build_video_fallback_script(reference: str, question: str) -> list:
    """降级视频脚本——以用户问题为核心，严肃专业。"""
    lines = [l.strip() for l in (reference or "").split("\n") if l.strip()]

    concepts = []
    for line in lines:
        cleaned = line.lstrip("#-*1234567890. ").strip()
        if len(cleaned) > 2:
            concepts.append(_safe_truncate(cleaned, 25))

    safe_title = _safe_truncate(question, 24)
    short_title = _safe_truncate(question, 12)

    # ── 封面 ──
    scenes = [{
        "scene": 1,
        "title": short_title,
        "slide_text": safe_title,
        "narration": f"本视频将解答以下问题：「{safe_title}」。以下将从基础概念入手，逐步展开讲解。"
    }]

    # ── 核心场景 ──
    scene_num = 2
    concept_narrations = [
        "第一个核心概念是{0}。它在整个体系中处于基础地位，后续内容均围绕其展开。掌握其定义与本质特征，是建立正确理解的前提。",
        "第二个要点是{0}。与前一概念相比，其关注维度不同，但二者之间存在紧密的逻辑关联。实际应用中，通常需要将二者结合分析。",
        "继续深入——{0}。此概念的难点在于实际场景中的准确运用，而非定义本身。建议结合具体案例加以理解。",
        "最后一个要点是{0}。它综合了前几个概念的核心思想，是从理论到实践的关键节点。",
    ]
    for i in range(min(len(concepts), 4)):
        c = concepts[i]
        scenes.append({
            "scene": scene_num,
            "title": _safe_truncate(c, 10),
            "slide_text": c,
            "narration": concept_narrations[i].format(c)
        })
        scene_num += 1

    if len(concepts) < 2:
        scenes.append({
            "scene": scene_num,
            "title": "深入分析",
            "slide_text": "原理与推导",
            "narration": "在掌握基本概念后，需要剖析其内在原理和推导过程。理解原理层面的逻辑，有助于形成系统化的分析思路。"
        })
        scene_num += 1

    # ── 总结 ──
    if len(concepts) >= 2:
        concat = "、".join(_safe_truncate(c, 6) for c in concepts[:4])
        summary = f"回顾本节内容。{concat}几个概念并非孤立存在，而是层层递进、相互支撑的关系。学习的价值在于理解概念之间的内在逻辑，从而实现知识的迁移与灵活运用。"
    else:
        summary = "知识体系的建立依赖逻辑梳理与反复实践。每一次深入理解，都是对认知边界的拓展。"

    scenes.append({
        "scene": scene_num,
        "title": "本节总结",
        "slide_text": "知识体系\n逻辑关联",
        "narration": summary
    })
    scene_num += 1

    # ── 结尾 ──
    scenes.append({
        "scene": scene_num,
        "title": "延伸学习",
        "slide_text": "课后练习\n拓展阅读",
        "narration": "本节讲解到此结束。建议通过练习检验理解程度，并结合拓展材料进一步深化。"
    })

    for i, s in enumerate(scenes):
        s["scene"] = i + 1
    return scenes


# ================== 画像合并 ==================

def merge_profile(existing: dict, new_profile: dict) -> dict:
    """合并新旧画像，数组字段取并集，标量字段智能覆盖。

    策略：
    - weak_points / error_patterns: 新旧取并集（去重），积累不丢失
    - 标量字段: 新值覆盖旧值（反映最新状态）
    - 避免画像降级：如果新值为空/默认值，保留已有值
    """
    merged = {**existing, **new_profile}

    # 数组字段取并集
    for arr_field in ("weak_points", "error_patterns"):
        if arr_field in new_profile and isinstance(new_profile[arr_field], list):
            existing_arr = existing.get(arr_field, [])
            if isinstance(existing_arr, list):
                merged[arr_field] = list(set(existing_arr + new_profile[arr_field]))

    # 标量字段: 如果新值为空或明显是兜底值，保留旧值
    non_empty_old = {k: v for k, v in existing.items() if v and v != "暂无" and v != "未知"}
    for key in ("learning_goals", "interest", "interaction_summary",
                "knowledge_base", "learning_style", "cognitive_style",
                "motivation_level"):
        new_val = merged.get(key, "")
        old_val = non_empty_old.get(key, "")
        if (not new_val or new_val in ("", "暂无", "未知", "--")) and old_val:
            merged[key] = old_val

    return merged


# ================== 豆包图片生成 ==================

def call_doubao_image(prompt: str, size: str = "2K") -> dict:
    """调用豆包(Seedream)图片生成 API。

    Args:
        prompt: 图片生成提示词
        size: 图片尺寸，默认 "2K"

    Returns:
        {"success": bool, "url": str, "error": str}
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DOUBAO_API_KEY}",
    }
    payload = {
        "model": DOUBAO_MODEL,
        "prompt": prompt,
        "sequential_image_generation": "disabled",
        "response_format": "url",
        "size": size,
        "stream": False,
        "watermark": True,
    }
    try:
        resp = requests.post(DOUBAO_API_URL, headers=headers, json=payload, timeout=120)
        if resp.status_code == 200:
            data = resp.json()
            # 提取图片 URL
            images = data.get("data", [])
            if images and len(images) > 0:
                url = images[0].get("url", "")
                if url:
                    logger.info("豆包图片生成成功: %s...", url[:80])
                    return {"success": True, "url": url, "error": ""}
            logger.warning("豆包图片生成返回空结果: %s", resp.text[:200])
            return {"success": False, "url": "", "error": "豆包 API 返回空结果"}
        else:
            logger.error("豆包图片生成失败 HTTP %d: %s", resp.status_code, resp.text[:300])
            return {"success": False, "url": "", "error": f"豆包 API 错误 ({resp.status_code}): {resp.text[:200]}"}
    except requests.exceptions.Timeout:
        logger.error("豆包图片生成超时")
        return {"success": False, "url": "", "error": "豆包 API 请求超时"}
    except Exception as e:
        logger.error("豆包图片生成异常: %s", e)
        return {"success": False, "url": "", "error": str(e)}
