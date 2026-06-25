#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
open-webSearch MCP 客户端
通过 MCP Streamable HTTP 协议调用本地 open-webSearch 服务进行联网搜索
"""

import requests
import json
import re
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

OPEN_WEBSEARCH_URL = "http://localhost:3000"

# MCP Streamable HTTP 要求的 Accept header
MCP_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
}


def _parse_sse(text: str) -> List[dict]:
    """解析 SSE (Server-Sent Events) 响应，提取 data 中的 JSON 对象"""
    results = []
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("data:"):
            data_str = line[5:].strip()
            if data_str:
                try:
                    results.append(json.loads(data_str))
                except json.JSONDecodeError:
                    logger.debug("SSE 解析跳过非 JSON: %.80s", data_str)
    return results


class OpenWebSearchClient:
    """MCP Streamable HTTP 客户端，与 open-webSearch 服务通信"""

    def __init__(self, base_url: str = OPEN_WEBSEARCH_URL):
        self.base_url = base_url
        self.mcp_url = f"{base_url}/mcp"
        self.session_id: Optional[str] = None
        self._request_id = 0

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def _post(self, payload: dict, timeout: int = 30) -> requests.Response:
        """发送 POST 请求到 MCP 端点"""
        headers = dict(MCP_HEADERS)
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id

        resp = requests.post(self.mcp_url, json=payload,
                            headers=headers, timeout=timeout)

        # 提取 session id
        sid = resp.headers.get("Mcp-Session-Id")
        if sid:
            self.session_id = sid

        return resp

    def _ensure_session(self):
        """确保 MCP 会话已初始化（含 initialized 通知）"""
        if self.session_id:
            return

        # 1. 发送 initialize 请求
        resp = self._post({
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "ai-learning-client", "version": "1.0.0"}
            },
            "id": self._next_id()
        })

        # 解析 initialize 响应 (可能是 SSE 或 JSON)
        content_type = resp.headers.get("content-type", "")
        if "text/event-stream" in content_type:
            _parse_sse(resp.text)  # 提取 session_id 已在 _post 中处理
        else:
            resp.json()

        if not self.session_id:
            raise RuntimeError("MCP 会话初始化失败: 未获取 session ID")

        # 2. 发送 initialized 通知 (无需响应)
        try:
            self._post({
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
                "params": {}
            }, timeout=5)
        except Exception:
            pass  # initialized 通知响应不重要

    def _rpc(self, method: str, params: dict = None,
             timeout: int = 30) -> dict:
        """发送 MCP JSON-RPC 请求，返回解析后的 result"""
        self._ensure_session()

        resp = self._post({
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
            "id": self._next_id()
        }, timeout=timeout)
        resp.raise_for_status()

        # 解析响应：SSE 或纯 JSON
        content_type = resp.headers.get("content-type", "")
        if "text/event-stream" in content_type:
            events = _parse_sse(resp.text)
            for evt in events:
                if "result" in evt:
                    return evt["result"]
                if "error" in evt:
                    logger.warning("MCP RPC 错误: %s", evt["error"])
                    return {}
            return {}
        else:
            data = resp.json()
            if "result" in data:
                return data["result"]
            if "error" in data:
                logger.warning("MCP RPC 错误: %s", data["error"])
            return {}

    def search(self, query: str, engines: List[str] = None,
               limit: int = 10, search_mode: str = "auto") -> dict:
        """调用 open-webSearch 的 search 工具，返回原始响应 dict"""
        if engines is None:
            engines = ["bing"]

        result = self._rpc("tools/call", {
            "name": "search",
            "arguments": {
                "query": query,
                "engines": engines,
                "limit": limit,
                "searchMode": search_mode
            }
        }, timeout=60)

        # 提取 content[0].text 中的 JSON
        content = result.get("content", [])
        if content and len(content) > 0:
            text = content[0].get("text", "{}")
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return {"results": [], "totalResults": 0, "raw": text}
        return {"results": [], "totalResults": 0, "engines": engines}

    def fetch_web(self, url: str, max_chars: int = 30000,
                  readability: bool = True, include_links: bool = False) -> dict:
        """获取网页内容"""
        result = self._rpc("tools/call", {
            "name": "fetchWebContent",
            "arguments": {
                "url": url,
                "maxChars": max_chars,
                "readability": readability,
                "includeLinks": include_links
            }
        }, timeout=30)

        content = result.get("content", [])
        if content and len(content) > 0:
            text = content[0].get("text", "{}")
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return {"raw": text}
        return {}

    def close(self):
        """关闭 MCP 会话"""
        if self.session_id:
            try:
                headers = {"Mcp-Session-Id": self.session_id}
                requests.delete(self.mcp_url, headers=headers, timeout=5)
            except Exception:
                pass
            self.session_id = None


# 全局单例
_client: Optional[OpenWebSearchClient] = None


def get_websearch_client() -> OpenWebSearchClient:
    """获取 open-webSearch 客户端单例"""
    global _client
    if _client is None:
        _client = OpenWebSearchClient()
    return _client


def search_open_websearch(query: str, count: int = 5,
                          engines: List[str] = None) -> List[Dict[str, str]]:
    """
    使用 open-webSearch 进行联网搜索。
    支持引擎: bing, baidu, duckduckgo, csdn, sogou, startpage, brave, exa 等。
    返回: [{title, url, snippet}, ...]
    """
    try:
        client = get_websearch_client()
        result = client.search(
            query,
            engines=engines or ["bing", "baidu"],
            limit=count
        )

        search_results = []
        for item in result.get("results", []):
            search_results.append({
                "title": item.get("title", ""),
                "url": item.get("url", item.get("link", "")),
                "snippet": item.get("snippet", item.get("description", "")),
            })

        logger.info("open-webSearch '%s' 返回 %d 条结果 (引擎: %s)",
                     query, len(search_results), result.get("engines", []))
        return search_results
    except requests.ConnectionError:
        logger.warning("open-webSearch 服务未运行 (端口 3000)")
    except Exception as e:
        logger.warning("open-webSearch 搜索异常: %s", e)
    return []
