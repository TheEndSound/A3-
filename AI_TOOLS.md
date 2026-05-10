# AI Coding Tools 使用说明

## Claude Code（Anthropic）

本项目使用 **Claude Code**（Anthropic 的 AI 编程助手）辅助开发。

### 使用方式

1. 在项目根目录下运行 `claude` 启动交互式会话
2. 以自然语言描述需求，Claude Code 自动读取、编辑代码
3. 支持多文件编辑、代码重构、调试等操作

### 使用场景

| 场景 | 说明 |
|------|------|
| 代码生成 | 根据需求描述生成完整的功能模块（智能体节点、API接口、前端界面） |
| 代码审查 | Claude Code 自动审查代码质量、发现潜在 bug |
| 重构优化 | 优化代码结构、提取公共逻辑（如 extract_json 函数） |
| 文档生成 | 自动生成项目文档 SYSTEM_DESIGN.md、TEST_REPORT.md 等 |
| 知识库构建 | 辅助构建《数据结构与算法》课程知识库 JSON |
| 多文件协同 | 同时在 graph_demo.py、graph_web.py、app.py 之间保持一致性 |

### 提示词示例

```
请为 graph_demo.py 中的 profile_agent 函数添加更鲁棒的 JSON 解析逻辑，
支持从模型输出中提取不规范的 JSON 对象。
```

### 效果评估

- 大幅提高了开发效率，尤其在多文件协同修改时
- 自动处理了 Python 包导入、类型注解等繁琐工作
- 提供了项目架构建议和最佳实践指导
- 辅助构建了完整的课程知识库（8章60+知识点）

## 其他 AI 工具

| 工具 | 用途 |
|------|------|
| DeepSeek 大模型 (deepseek-chat) | 系统核心 AI 能力，驱动所有智能体节点 |
