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
from web_search_client import search_open_websearch

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

# 新闻/资讯类关键词 — 说明用户在找实时信息而非学习内容
_NEWS_KEYWORDS = [
    "新闻", "热点", "最新", "今天", "昨天", "前天", "后天",
    "近日", "本周", "最近发生", "大事", "动态", "资讯", "消息", "报道",
    "热搜", "头条", "趋势", "实时", "刚刚",
]

# 搜索噪声词 — 从查询中移除（会干扰搜索引擎的动词/虚词）
_SEARCH_NOISE = ["有什么", "有哪些", "帮我查", "帮我搜", "帮我找", "帮我",
                  "我想知道", "我想了解", "请问", "请告诉我",
                  "搜索一下", "查找一下", "找一下", "查一下",
                  "搜索", "查找", "获取", "查询", "我想查",
                  "是什么", "什么是", "有没有", "能不能",
                  "告诉我", "的"]

# 纯日期模式 — 用户直接给了日期，不需要再搜索确认
_DATE_PATTERN = r'\d{4}年\d{1,2}月\d{1,2}日'


def _is_news_query(query: str) -> bool:
    """判断用户是否在查询新闻/实时信息"""
    return any(kw in query for kw in _NEWS_KEYWORDS)


def optimize_search_query(query: str) -> str:
    """
    优化搜索查询，提升搜索精度：
    - 检测新闻/实时类查询，附加日期上下文
    - 去除噪声词，提炼核心关键词
    - 中文时间词 → 具体日期
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

    # 新闻/实时类查询：附加日期避免歧义
    if _is_news_query(query):
        today = datetime.date.today()
        date_tag = today.strftime("%Y年%m月%d日")
        # 替换中文时间词为具体日期，避免搜到同名影视作品
        optimized = optimized.replace("前天",
                                       (today - datetime.timedelta(days=2)).strftime("%Y年%m月%d日"))
        optimized = optimized.replace("昨天",
                                       (today - datetime.timedelta(days=1)).strftime("%Y年%m月%d日"))
        optimized = optimized.replace("今天", date_tag)
        optimized = optimized.replace("明天",
                                       (today + datetime.timedelta(days=1)).strftime("%Y年%m月%d日"))
        optimized = optimized.replace("后天",
                                       (today + datetime.timedelta(days=2)).strftime("%Y年%m月%d日"))
        # 追加"新闻"确保搜索引擎理解意图
        if "新闻" not in optimized:
            optimized = f"{optimized} 新闻"

    # 纯日期查询（日期后没有具体主题词）→ 补充搜索语境
    _TOPIC_WORDS = ["新闻", "热点", "资讯", "动态", "科技", "财经", "体育", "娱乐", "要闻"]
    has_topic = any(tw in optimized for tw in _TOPIC_WORDS)
    if re.search(_DATE_PATTERN, optimized) and len(optimized) < 20 and not has_topic:
        optimized = f"{optimized} 要闻 热点"

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


def format_search_context(results: List[Dict[str, str]],
                          user_query: str = "") -> str:
    """将搜索结果格式化为注入 LLM 的上下文字符串。
    根据用户原始查询判断意图，给出针对性的指令。"""
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
        # 新闻/资讯模式：要求简洁摘要，不要展开教学
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
        # 学习模式：允许展开讲解，但基于事实
        lines += [
            "⚠️ 以下是搜索引擎返回的真实信息。你的回答必须：",
            "  1. 只陈述搜索结果中实际存在的内容",
            "  2. 引用具体来源",
            "  3. 如果搜索结果不包含用户问题的答案，直接说「搜索结果中未找到相关信息」，禁止编造",
            "  4. 禁止编造任何数字、日期、人名、事件名，除非搜索结果中明确提到",
            "",
        ]

    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r['title']}")
        lines.append(f"   {r['snippet']}")
        lines.append(f"   来源: {r['url']}")
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
  注意：如果用户是在查询新闻、热点、资讯、实时动态（如"今天有什么新闻"、"昨天热点"），这不是resource，应判为chat。
- tutor: 请求辅导或解释，包含"为什么"、"我不懂"、"帮我解释"、"什么意思"、"怎么理解"、"讲解一下"、"辅导我"、"帮我分析"等疑问
- evaluation: 请求学习评估，包含"评估"、"测试我"、"考核"、"我的学习效果"、"检测"等关键词
- chat: 普通学术问题、闲聊、新闻资讯查询、实时信息查询
输入: {last_message}
输出:"""


def build_chat_system_prompt(profile: Optional[dict], course_ctx: str) -> str:
    """构建普通对话的系统 prompt。
    当搜索结果中包含新闻/资讯指令时，自动切换为信息摘要模式。"""
    style_guide = get_style_guide(profile)
    has_search = "[联网搜索结果" in course_ctx
    is_news_mode = "信息摘要助手" in course_ctx
    search_note = ""
    if has_search:
        if is_news_mode:
            search_note = (
                "\n⚠️ 重要：你当前处于**信息摘要模式**。"
                "\n你的任务是对搜索结果进行简洁归纳，不是生成教学材料。"
                "\n禁止生成：讲解文档、练习题、思维导图、视频脚本、实操案例等教学格式。\n"
            )
        else:
            search_note = (
                "\n⚠️ 重要：上述 [联网搜索结果] 是从搜索引擎获取的**实时真实数据**。"
                "\n你必须：1) 优先基于搜索结果回答 2) 引用搜索结果的标题和来源 "
                "3) 如果搜索结果不够充分，请诚实说明。禁止编造不存在的事实。\n"
            )

    if is_news_mode:
        return f"""你是一个智能信息摘要助手「AI智学」，帮助用户快速了解实时资讯。
请根据以下学生画像调整回答方式：
- 知识基础：{profile.get('knowledge_base', '初级') if profile else '初级'}
- 学习风格：{profile.get('learning_style', '视觉型') if profile else '视觉型'}，{style_guide}
- 兴趣方向：{profile.get('interest', '编程') if profile else '编程'}

{course_ctx}
{search_note}
请简洁、结构化地呈现信息。用条目式输出，每条新闻配一句话摘要+来源链接。"""

    return f"""你是一个智能学习助手「AI智学」，提供个性化学习服务。
请根据以下学生画像调整回答方式：
- 知识基础：{profile.get('knowledge_base', '初级') if profile else '初级'}
- 学习风格：{profile.get('learning_style', '视觉型') if profile else '视觉型'}，{style_guide}
- 薄弱点：{', '.join(profile.get('weak_points', ['无'])) if profile else '无'}
- 兴趣方向：{profile.get('interest', '编程') if profile else '编程'}

{course_ctx}
{search_note}
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
