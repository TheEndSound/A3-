#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
B站视频下载模块 — 封装爬虫功能供 Web 应用调用
"""

import os
import re
import json
import time
import subprocess
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.bilibili.com",
}

API_VIEW = "https://api.bilibili.com/x/web-interface/view"
API_PLAYURL = "https://api.bilibili.com/x/player/playurl"

DEFAULT_QN = 116
THREAD_COUNT = 4
MAX_RETRIES = 3
RETRY_DELAY = 2

QN_MAP = {
    127: "8K", 125: "HDR", 120: "4K",
    116: "1080P60", 112: "1080P+", 80: "1080P",
    74: "720P60", 64: "720P", 32: "480P", 16: "360P",
}


def _api_request(session, url, params=None, retries=MAX_RETRIES):
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(url, params=params, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if data.get("code") != 0:
                raise RuntimeError(f"API 错误: code={data.get('code')}, message={data.get('message')}")
            return data
        except requests.exceptions.RequestException as e:
            last_exc = e
            if attempt < retries:
                time.sleep(RETRY_DELAY)
    raise RuntimeError(f"API 请求失败: {last_exc}")


def _get_video_info(session, bvid):
    data = _api_request(session, API_VIEW, params={"bvid": bvid})
    return data["data"]


def _get_play_url(session, bvid, cid, qn):
    params = {"bvid": bvid, "cid": cid, "qn": qn, "fnval": 4048, "fourk": 1}
    data = _api_request(session, API_PLAYURL, params=params)
    return data["data"]


def _build_full_url(raw_url):
    raw_url = raw_url.strip()
    if raw_url.startswith("http://") or raw_url.startswith("https://"):
        return raw_url
    if raw_url.startswith("//"):
        return "https:" + raw_url
    return "https://" + raw_url


def _probe_file_size(session, url):
    for method in ("head", "get"):
        try:
            if method == "head":
                resp = session.head(url, headers=HEADERS, timeout=30)
            else:
                resp = session.get(url, headers=HEADERS, stream=True, timeout=30)
                resp.close()
            if 200 <= resp.status_code < 300:
                size = int(resp.headers.get("content-length", 0))
                ar = resp.headers.get("accept-ranges", "")
                supports = "bytes" in ar.lower()
                return size, supports
        except Exception:
            continue
    raise RuntimeError(f"无法探测文件信息: {url}")


def _download_range(session, url, start, end, filepath, timeout=60):
    range_header = f"bytes={start}-{end - 1}"
    resp = session.get(url, headers={**HEADERS, "Range": range_header}, stream=True, timeout=timeout)
    resp.raise_for_status()
    with open(filepath, "r+b") as f:
        f.seek(start)
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)


def _download_file(session, url, filepath, progress_callback=None):
    url = _build_full_url(url)
    part_path = Path(filepath).with_suffix(Path(filepath).suffix + ".part")

    try:
        total_size, supports_range = _probe_file_size(session, url)
    except Exception:
        total_size, supports_range = 0, False

    if total_size == 0 or not supports_range:
        resp = session.get(url, headers=HEADERS, stream=True, timeout=60)
        resp.raise_for_status()
        Path(part_path).parent.mkdir(parents=True, exist_ok=True)
        downloaded = 0
        with open(part_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress_callback:
                        progress_callback(downloaded, total_size or downloaded)
        Path(part_path).rename(filepath)
        if progress_callback:
            progress_callback(total_size or downloaded, total_size or downloaded)
        return

    Path(part_path).parent.mkdir(parents=True, exist_ok=True)
    with open(part_path, "wb") as f:
        f.truncate(total_size)

    ranges = []
    for i in range(THREAD_COUNT):
        s = i * (total_size // THREAD_COUNT)
        e = total_size if i == THREAD_COUNT - 1 else (i + 1) * (total_size // THREAD_COUNT)
        if s < e:
            ranges.append((s, e))

    downloaded_bytes = [0]
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=THREAD_COUNT) as executor:
        futures = {}
        for start, end in ranges:
            future = executor.submit(_download_range, session, url, start, end, str(part_path))
            futures[future] = (start, end)

        for future in as_completed(futures):
            start, end = futures[future]
            try:
                future.result()
            except Exception as e:
                raise RuntimeError(f"分块下载失败 [{start}-{end}): {e}")
            with lock:
                downloaded_bytes[0] += (end - start)
                if progress_callback:
                    progress_callback(downloaded_bytes[0], total_size)

    if Path(filepath).exists():
        Path(filepath).unlink()
    Path(part_path).rename(filepath)


def _find_ffmpeg():
    import shutil
    found = shutil.which("ffmpeg")
    if found:
        return found
    search_paths = [
        "C:\\ffmpeg\\bin",
        os.path.expandvars(r"%LOCALAPPDATA%\\JianyingPro"),
        os.path.expandvars(r"%PROGRAMFILES%\\ffmpeg\\bin"),
    ]
    for base in search_paths:
        base_path = Path(base)
        if base_path.is_dir():
            for candidate in base_path.rglob("ffmpeg.exe"):
                return str(candidate)
    return None


def _merge_audio_video(video_path, audio_path, output_path):
    ffmpeg_path = _find_ffmpeg()
    if not ffmpeg_path:
        raise RuntimeError("未找到 ffmpeg，无法合并音视频。请安装 ffmpeg 后重试。")
    cmd = [
        ffmpeg_path, "-y",
        "-i", str(video_path), "-i", str(audio_path),
        "-c", "copy", "-movflags", "+faststart",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        cmd2 = [
            ffmpeg_path, "-y",
            "-i", str(video_path), "-i", str(audio_path),
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            str(output_path),
        ]
        subprocess.run(cmd2, check=True, encoding="utf-8", errors="replace",
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _select_best_streams(dash):
    video_streams = dash.get("video", [])
    audio_streams = dash.get("audio", [])
    if not video_streams:
        raise RuntimeError("未获取到任何视频流")
    if not audio_streams:
        raise RuntimeError("未获取到任何音频流")
    h264_videos = [s for s in video_streams if s.get("codecid") == 7 or "avc1" in s.get("codecs", "")]
    candidates = h264_videos if h264_videos else video_streams
    candidates.sort(key=lambda s: s.get("bandwidth", 0), reverse=True)
    audio_streams.sort(key=lambda s: s.get("bandwidth", 0), reverse=True)
    return candidates[0], audio_streams[0]


# ================== 公开接口 ==================

_download_tasks: dict = {}
_tasks_lock = threading.Lock()


def get_video_info_simple(bvid: str) -> dict:
    """获取视频基本信息（标题、封面、分P等）。"""
    session = requests.Session()
    info = _get_video_info(session, bvid)
    return {
        "bvid": bvid,
        "title": info.get("title", ""),
        "pic": info.get("pic", ""),
        "author": info.get("owner", {}).get("name", ""),
        "duration": info.get("duration", 0),
        "pages": len(info.get("pages", [info])),
        "desc": info.get("desc", "")[:200],
    }


def download_video_async(bvid: str, output_dir: str = None, qn: int = DEFAULT_QN) -> str:
    """异步下载B站视频，返回 task_id 用于查询进度。"""
    import uuid
    task_id = str(uuid.uuid4())[:8]

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "downloads")

    with _tasks_lock:
        _download_tasks[task_id] = {
            "status": "pending",
            "progress": 0,
            "bvid": bvid,
            "filepath": None,
            "error": None,
        }

    def _run():
        try:
            with _tasks_lock:
                _download_tasks[task_id]["status"] = "running"
            session = requests.Session()
            video_info = _get_video_info(session, bvid)
            pages = video_info.get("pages", [video_info])
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            for idx, page in enumerate(pages):
                page_cid = page["cid"]
                page_title = page.get("part", f"P{idx + 1}")
                page_index = idx + 1 if len(pages) > 1 else 0

                safe_title = re.sub(r'[\\/:*?"<>|]', "_", page_title)
                suffix = f"_P{page_index}" if page_index else ""

                play_data = _get_play_url(session, bvid, page_cid, qn)
                dash = play_data.get("dash")
                if not dash:
                    raise RuntimeError("该视频未提供 DASH 格式")

                video_stream, audio_stream = _select_best_streams(dash)

                video_temp = output_path / f"{safe_title}{suffix}_video.m4s"
                audio_temp = output_path / f"{safe_title}{suffix}_audio.m4s"
                final_output = output_path / f"{safe_title}{suffix}.mp4"

                video_url = video_stream.get("base_url") or video_stream.get("baseUrl") or video_stream.get("url")
                audio_url = audio_stream.get("base_url") or audio_stream.get("baseUrl") or audio_stream.get("url")

                if not video_url or not audio_url:
                    raise RuntimeError("未能提取有效的流地址")

                def progress_cb(done, total):
                    with _tasks_lock:
                        if task_id in _download_tasks:
                            _download_tasks[task_id]["progress"] = int(done / total * 100) if total else 0

                _download_file(session, video_url, str(video_temp), progress_callback=progress_cb)
                _download_file(session, audio_url, str(audio_temp), progress_callback=progress_cb)

                _merge_audio_video(video_temp, audio_temp, final_output)

                video_temp.unlink(missing_ok=True)
                audio_temp.unlink(missing_ok=True)

                with _tasks_lock:
                    if task_id in _download_tasks:
                        _download_tasks[task_id]["status"] = "done"
                        _download_tasks[task_id]["progress"] = 100
                        _download_tasks[task_id]["filepath"] = str(final_output)
                return

        except Exception as e:
            with _tasks_lock:
                if task_id in _download_tasks:
                    _download_tasks[task_id]["status"] = "error"
                    _download_tasks[task_id]["error"] = str(e)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return task_id


def get_download_progress(task_id: str) -> dict:
    """查询下载进度。"""
    with _tasks_lock:
        return _download_tasks.get(task_id, {"status": "not_found"})
