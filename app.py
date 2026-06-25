#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Web版个性化学习多智能体系统
FastAPI + SSE + 完整多智能体工作流 + 精美UI
"""

import os
import io as _io
import json
import re as _re
import asyncio
import subprocess
import tempfile
from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse
from dotenv import load_dotenv

from docx import Document as _Document
from docx.shared import Inches as _Inches, Pt as _Pt, RGBColor as _RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH as _WD_ALIGN
from docx.oxml.ns import qn as _qn

try:
    from pypdf import PdfReader as _PdfReader
except ImportError:
    _PdfReader = None

from graph_web import process_message
from video_downloader import download_video_async, get_download_progress, get_video_info_simple
from database import list_sessions, get_session, get_session_messages, delete_session, \
    insert_document, list_documents, get_document, delete_document, upsert_session

load_dotenv()

app = FastAPI(title="AI智学 - 个性化学习多智能体系统")

# 挂载静态文件目录
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# 从 index.html 加载前端页面
_INDEX_HTML = None


def _load_index_html() -> str:
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<html><body><h1>前端页面丢失</h1></body></html>"


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(
        content=_load_index_html(),
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


def _decode_base64_file(content_b64: str, ext: str) -> str:
    """将前端发来的 base64 文件内容解码并解析为文本。"""
    import base64
    try:
        raw = base64.b64decode(content_b64)
    except Exception:
        raise HTTPException(status_code=400, detail="文件 base64 解码失败")
    if ext == ".pdf":
        return _parse_pdf(raw)
    elif ext == ".docx":
        return _parse_docx(raw)
    elif ext == ".doc":
        return _parse_doc(raw)
    else:
        # fallback: try UTF-8
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            try:
                return raw.decode("gbk")
            except UnicodeDecodeError:
                raise HTTPException(status_code=400, detail="无法解码文件内容")


def _build_file_context(files: list) -> str:
    """将前端 files 数组格式化为消息前缀。"""
    lines = ["[用户上传了以下文件]\n"]
    for i, f in enumerate(files):
        fname = f.get("name", f"file_{i}")
        ftype = f.get("type", "text")
        ext = f.get("ext", "")
        content = f.get("content", "")
        if ftype == "image":
            lines.append(f"📷 图片 {fname}: data:{f.get('mime','image/png')};base64,{content[:80]}...")
        elif ftype == "text":
            lines.append(f"📄 文件 {fname}:\n```\n{content}\n```")
        elif ftype == "binary":
            parsed = _decode_base64_file(content, ext)
            lines.append(f"📄 文件 {fname}:\n```\n{parsed}\n```")
    lines.append("[文件内容结束]\n")
    return "\n".join(lines)


@app.post("/chat/stream")
async def chat_stream(request: Request):
    data = await request.json()
    user_message = data.get("message", "")
    session_id = data.get("session_id", "default")
    image_mode = data.get("image_mode", False)
    web_search = data.get("web_search", False)
    deep_thinking = data.get("deep_thinking", False)
    files = data.get("files", [])
    print(f"[DEBUG chat_stream] web_search={web_search} image_mode={image_mode} deep_thinking={deep_thinking} files={len(files)} msg={user_message[:40]}", flush=True)

    # 输入验证
    if (not user_message or not user_message.strip()) and not files:
        async def empty_response():
            yield {"data": json.dumps({"type": "status", "content": "⚠️ 消息不能为空，请输入学习相关问题。"})}
            yield {"data": json.dumps({"type": "end"})}
        return EventSourceResponse(empty_response())
    if len(user_message) > 2000:
        async def too_long_response():
            yield {"data": json.dumps({"type": "status", "content": "⚠️ 消息过长（超过2000字符），请精简后重试。"})}
            yield {"data": json.dumps({"type": "end"})}
        return EventSourceResponse(too_long_response())
    if len(session_id) > 128:
        async def bad_session_response():
            yield {"data": json.dumps({"type": "status", "content": "⚠️ session_id 格式无效。"})}
            yield {"data": json.dumps({"type": "end"})}
        return EventSourceResponse(bad_session_response())

    # 构建多模态消息
    full_message = user_message
    if files:
        file_context = _build_file_context(files)
        full_message = file_context + "\n" + (user_message or "请帮我分析上传的文件内容")

    async def event_generator():
        gen = process_message(session_id, full_message, image_mode=image_mode, web_search=web_search, deep_thinking=deep_thinking)
        ended = False
        try:
            for event_type, content in gen:
                if await request.is_disconnected():
                    # 关闭 generator 以触发其 finally 块（持久化AI消息）
                    gen.close()
                    break
                yield {"data": json.dumps({"type": event_type, "content": content})}
                if event_type == "end":
                    ended = True
        except asyncio.CancelledError:
            gen.close()
        except Exception as e:
            gen.close()
            yield {"data": json.dumps({"type": "error", "content": str(e)})}
        finally:
            if not ended:
                yield {"data": json.dumps({"type": "end"})}

    return EventSourceResponse(event_generator())


@app.post("/video/download")
async def start_video_download(request: Request):
    data = await request.json()
    bvid = data.get("bvid", "").strip()
    if not bvid:
        raise HTTPException(status_code=400, detail="请提供视频 BV 号")
    task_id = download_video_async(bvid)
    return {"task_id": task_id, "status": "started"}


@app.get("/video/download/{task_id}")
async def check_video_download(task_id: str):
    progress = get_download_progress(task_id)
    if progress.get("status") == "not_found":
        raise HTTPException(status_code=404, detail="任务不存在或已过期")
    return progress


@app.get("/video/info/{bvid}")
async def video_info(bvid: str):
    try:
        info = get_video_info_simple(bvid)
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions")
async def api_list_sessions(limit: int = 20):
    return list_sessions(limit=limit)


@app.get("/sessions/{session_id}/messages")
async def api_get_session_messages(session_id: str):
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    messages = get_session_messages(session_id)
    return {"session": session, "messages": messages}


@app.delete("/sessions/{session_id}")
async def api_delete_session(session_id: str):
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    delete_session(session_id)
    return {"status": "deleted", "session_id": session_id}


ALLOWED_EXTENSIONS = {
    ".txt", ".md", ".json", ".csv", ".html", ".py", ".js", ".css", ".xml", ".yaml", ".log",
    ".pdf", ".docx", ".doc",
}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB


def _parse_pdf(raw: bytes) -> str:
    """从 PDF 字节数据提取文本。"""
    if _PdfReader is None:
        raise HTTPException(status_code=500, detail="pypdf 库未安装，无法解析 PDF")
    try:
        reader = _PdfReader(_io.BytesIO(raw))
        parts = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                parts.append(text)
        return "\n\n".join(parts)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF 解析失败: {str(e)}")


def _parse_doc(raw: bytes) -> str:
    """通过 Word COM 自动化从旧版 .doc 提取文本。"""
    try:
        import pythoncom
        from win32com.client import Dispatch
    except ImportError:
        raise HTTPException(status_code=500, detail="pywin32 库未安装，无法解析 .doc 文件，请转换为 .docx 格式后重试")

    # 将 bytes 写入临时文件（COM 需要文件路径）
    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(suffix=".doc")
        os.close(fd)
        with open(tmp_path, "wb") as f:
            f.write(raw)

        pythoncom.CoInitialize()
        word = Dispatch("Word.Application")
        word.Visible = False
        word.DisplayAlerts = 0

        doc = word.Documents.Open(tmp_path)
        text = doc.Content.Text

        doc.Close(False)
        word.Quit()
        pythoncom.CoUninitialize()

        if not text or not text.strip():
            raise HTTPException(status_code=400, detail=".doc 文件无法提取文本，请转换为 .docx 格式后重试")
        return text
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f".doc 解析失败: {str(e)}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


def _parse_docx(raw: bytes) -> str:
    """从 DOCX 字节数据提取文本。"""
    if _Document is None:
        raise HTTPException(status_code=500, detail="python-docx 库未安装，无法解析 DOCX")
    try:
        doc = _Document(_io.BytesIO(raw))
        parts = []
        for para in doc.paragraphs:
            if para.text.strip():
                parts.append(para.text)
        return "\n".join(parts)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"DOCX 解析失败: {str(e)}")


@app.post("/knowledge/upload")
async def upload_knowledge_file(session_id: str = Form(...), file: UploadFile = File(...)):
    """上传文件导入知识库，支持 txt/md/pdf/docx 等格式。"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="未选择文件")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"不支持的文件类型: {ext}。支持: {', '.join(sorted(ALLOWED_EXTENSIONS))}")

    try:
        raw = await file.read()
    except Exception:
        raise HTTPException(status_code=400, detail="读取文件失败")

    if len(raw) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail=f"文件过大（最大 10MB），当前 {len(raw) / 1024 / 1024:.1f}MB")

    # 根据文件类型解析
    if ext == ".pdf":
        content = _parse_pdf(raw)
    elif ext == ".docx":
        content = _parse_docx(raw)
    elif ext == ".doc":
        content = _parse_doc(raw)
    else:
        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError:
            try:
                content = raw.decode("gbk")
            except UnicodeDecodeError:
                raise HTTPException(status_code=400, detail="无法解码文件内容，请使用 UTF-8 或 GBK 编码")

    # 生成预览（前 200 个字符）
    preview = content[:200].replace("\n", " ").strip()
    if len(content) > 200:
        preview += "..."

    content_type = ext.lstrip(".")
    upsert_session(session_id)  # 确保 session 存在以满足外键约束
    doc_id = insert_document(session_id, file.filename, content, content_type, len(raw))
    return {
        "status": "ok",
        "doc_id": doc_id,
        "title": file.filename,
        "size": len(raw),
        "preview": preview,
        "chars": len(content),
    }


@app.post("/knowledge/import-text")
async def import_knowledge_text(request: Request):
    """粘贴文本导入知识库。"""
    data = await request.json()
    session_id = data.get("session_id", "").strip()
    title = data.get("title", "").strip()
    content = data.get("content", "")

    if not session_id:
        raise HTTPException(status_code=400, detail="缺少 session_id")
    if not title:
        raise HTTPException(status_code=400, detail="请输入文档标题")
    if not content or not content.strip():
        raise HTTPException(status_code=400, detail="请输入文档内容")
    if len(content) > 2 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="内容过长（最大 2MB）")

    upsert_session(session_id)
    doc_id = insert_document(session_id, title, content, "text", len(content.encode("utf-8")))
    return {"status": "ok", "doc_id": doc_id, "title": title}


@app.get("/knowledge/{session_id}")
async def api_list_knowledge(session_id: str):
    """列出会话的所有知识库文档。"""
    return list_documents(session_id)


@app.delete("/knowledge/{doc_id}")
async def api_delete_knowledge(doc_id: int):
    """删除指定知识库文档。"""
    doc = get_document(doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="文档不存在")
    delete_document(doc_id)
    return {"status": "deleted", "doc_id": doc_id}


@app.post("/resource/export-docx")
async def export_resource_docx(request: Request):
    """将 Markdown 资源内容转换为 Word 文档并下载。"""
    data = await request.json()
    markdown = data.get("content", "")
    title = data.get("title", "学习资源")

    if not markdown or not markdown.strip():
        raise HTTPException(status_code=400, detail="内容为空，无法生成文档")

    doc = _Document()

    # 设置默认字体
    style = doc.styles["Normal"]
    font = style.font
    font.name = "Microsoft YaHei"
    font.size = _Pt(11)
    style.element.rPr.rFonts.set(_qn("w:eastAsia"), "微软雅黑")

    # 设置页边距
    for section in doc.sections:
        section.top_margin = _Inches(0.8)
        section.bottom_margin = _Inches(0.8)
        section.left_margin = _Inches(1.0)
        section.right_margin = _Inches(1.0)

    # 文档标题
    doc_title = doc.add_heading(title, level=0)
    doc_title.alignment = _WD_ALIGN.CENTER

    lines = markdown.split("\n")
    i = 0
    in_code_block = False
    code_lang = ""
    code_lines = []

    def add_formatted_paragraph(text, style_name=None):
        """添加带格式化（粗体、斜体、行内代码）的段落。"""
        p = doc.add_paragraph()
        if style_name:
            p.style = style_name
        _parse_inline(p, text)
        return p

    def _parse_inline(paragraph, text):
        """解析行内格式：**粗体**、*斜体*、`代码`。"""
        # 正则匹配行内格式
        pattern = r"(\*\*(.+?)\*\*|\*(.+?)\*|`([^`]+)`|\[([^\]]+)\]\(([^)]+)\))"
        last_end = 0
        for m in _re.finditer(pattern, text):
            # 添加前面的纯文本
            if m.start() > last_end:
                paragraph.add_run(text[last_end:m.start()])
            if m.group(2):  # **粗体**
                run = paragraph.add_run(m.group(2))
                run.bold = True
            elif m.group(3):  # *斜体*
                run = paragraph.add_run(m.group(3))
                run.italic = True
            elif m.group(4):  # `行内代码`
                run = paragraph.add_run(m.group(4))
                run.font.name = "Consolas"
                run.font.size = _Pt(10)
            elif m.group(5):  # [链接](url)
                run = paragraph.add_run(m.group(5))
                run.underline = True
                run.font.color.rgb = _RGBColor(37, 99, 235)
            last_end = m.end()
        # 添加剩余纯文本
        if last_end < len(text):
            paragraph.add_run(text[last_end:])

    def add_code_block(lang, code):
        """添加代码块。"""
        if lang:
            p = doc.add_paragraph()
            run = p.add_run(f"语言: {lang}")
            run.font.size = _Pt(9)
            run.font.color.rgb = _RGBColor(100, 100, 100)
            run.italic = True
        for line in code:
            p = doc.add_paragraph()
            p.style = doc.styles["No Spacing"] if "No Spacing" in [s.name for s in doc.styles] else doc.styles["Normal"]
            p.paragraph_format.space_before = _Pt(0)
            p.paragraph_format.space_after = _Pt(0)
            run = p.add_run(line)
            run.font.name = "Consolas"
            run.font.size = _Pt(9.5)
        doc.add_paragraph()  # 空行分隔

    def add_table_block(rows_data):
        """添加表格。"""
        if not rows_data or len(rows_data) < 2:
            return
        num_cols = max(len(row) for row in rows_data)
        table = doc.add_table(rows=len(rows_data), cols=num_cols)
        table.style = "Light Grid Accent 1"
        for r_idx, row in enumerate(rows_data):
            for c_idx, cell_text in enumerate(row):
                if c_idx < num_cols:
                    cell = table.cell(r_idx, c_idx)
                    cell.text = cell_text.strip()
                    for para in cell.paragraphs:
                        para.paragraph_format.space_before = _Pt(2)
                        para.paragraph_format.space_after = _Pt(2)
                        for run in para.runs:
                            run.font.size = _Pt(10)
                            if r_idx == 0:
                                run.bold = True
        doc.add_paragraph()  # 空行分隔

    while i < len(lines):
        line = lines[i]

        # 代码块开始
        m_code_start = _re.match(r"^```(\w*)$", line)
        if m_code_start and not in_code_block:
            in_code_block = True
            code_lang = m_code_start.group(1) or ""
            code_lines = []
            i += 1
            continue

        # 代码块结束
        if line.strip() == "```" and in_code_block:
            add_code_block(code_lang, code_lines)
            in_code_block = False
            code_lang = ""
            code_lines = []
            i += 1
            continue

        # 代码块内容
        if in_code_block:
            code_lines.append(line)
            i += 1
            continue

        # 空行
        if not line.strip():
            i += 1
            continue

        # 表格（| ... |）
        if "|" in line and line.strip().startswith("|"):
            table_rows = []
            while i < len(lines) and "|" in lines[i] and lines[i].strip().startswith("|"):
                row_line = lines[i].strip()
                if not _re.match(r"^\|[\s\-:|]+\|$", row_line):  # 跳过分隔行
                    cells = [c.strip() for c in row_line.split("|")[1:-1]]
                    if cells:
                        table_rows.append(cells)
                i += 1
            if table_rows:
                add_table_block(table_rows)
            continue

        # 标题
        heading_match = _re.match(r"^(#{1,4})\s+(.+)$", line)
        if heading_match:
            level = min(len(heading_match.group(1)), 3)  # Word 支持1-3级，映射#->1, ##->2, ###->3
            text = heading_match.group(2).strip()
            # 清除标题中的 markdown 标记
            text = _re.sub(r"\*\*(.+?)\*\*", r"\1", text)
            text = _re.sub(r"\*(.+?)\*", r"\1", text)
            text = _re.sub(r"`(.+?)`", r"\1", text)
            doc.add_heading(text, level=level)
            i += 1
            continue

        # 水平线
        if _re.match(r"^---+$", line.strip()):
            doc.add_paragraph("─" * 50)
            i += 1
            continue

        # 无序列表
        ul_match = _re.match(r"^(\s*)[-*]\s+(.+)$", line)
        if ul_match:
            text = ul_match.group(2)
            p = doc.add_paragraph(style="List Bullet")
            p.clear()
            _parse_inline(p, text)
            i += 1
            continue

        # 有序列表
        ol_match = _re.match(r"^(\s*)\d+\.\s+(.+)$", line)
        if ol_match:
            text = ol_match.group(2)
            p = doc.add_paragraph(style="List Number")
            p.clear()
            _parse_inline(p, text)
            i += 1
            continue

        # 普通段落
        add_formatted_paragraph(line.strip())
        i += 1

    # 保存到内存
    buf = _io.BytesIO()
    doc.save(buf)
    buf.seek(0)

    safe_filename = _re.sub(r'[\\/*?:"<>|]', "_", title)[:50]
    return Response(
        content=buf.getvalue(),
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f'attachment; filename="{safe_filename}.docx"'}
    )


@app.post("/api/render-image")
async def render_mermaid_image(request: Request):
    """服务端渲染 Mermaid 代码为 JPG 图片。"""
    data = await request.json()
    code = data.get("code", "").strip()
    if not code:
        raise HTTPException(status_code=400, detail="缺少 mermaid 代码")
    if len(code) > 10000:
        raise HTTPException(status_code=400, detail="代码过长")

    script = os.path.join(os.path.dirname(__file__), "render_mermaid.cjs")
    try:
        proc = await asyncio.create_subprocess_exec(
            "node", script,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(input=code.encode("utf-8")), timeout=30
        )
        if proc.returncode != 0:
            err_msg = stderr.decode("utf-8", errors="replace")[:200]
            raise HTTPException(status_code=500, detail=f"渲染失败: {err_msg}")
        svg = stdout.decode("utf-8", errors="replace")

        # SVG → PNG → JPG 转换
        import cairosvg
        from PIL import Image

        png_data = cairosvg.svg2png(bytestring=svg.encode("utf-8"))
        img = Image.open(_io.BytesIO(png_data))
        if img.mode in ("RGBA", "P"):
            bg = Image.new("RGB", img.size, (255, 255, 255))
            if img.mode == "RGBA":
                bg.paste(img, mask=img.split()[3])
            else:
                bg.paste(img)
            img = bg
        elif img.mode != "RGB":
            img = img.convert("RGB")

        jpg_buf = _io.BytesIO()
        img.save(jpg_buf, format="JPEG", quality=90)
        jpg_buf.seek(0)
        return Response(content=jpg_buf.getvalue(), media_type="image/jpeg")
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="渲染超时")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
