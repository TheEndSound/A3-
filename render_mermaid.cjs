// 服务端 Mermaid 渲染脚本 (Node.js)
// 从 stdin 读取 mermaid 代码，输出 SVG
const { JSDOM } = require('jsdom');

// 创建虚拟 DOM 环境供 mermaid 使用
const dom = new JSDOM('<!DOCTYPE html><html><body></body></html>', { url: 'http://localhost' });
global.window = dom.window;
global.document = dom.window.document;
global.navigator = dom.window.navigator;
global.XMLSerializer = dom.window.XMLSerializer;
global.CSSStyleSheet = dom.window.CSSStyleSheet;

const mermaid = require('mermaid').default;

// 初始化 mermaid
mermaid.initialize({ startOnLoad: false, securityLevel: 'loose' });

let code = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', function(chunk) { code += chunk; });
process.stdin.on('end', async function() {
    try {
        const id = 'diagram-' + Date.now();
        const { svg } = await mermaid.render(id, code.trim());
        process.stdout.write(svg);
    } catch (e) {
        process.stderr.write(e.message || String(e));
        process.exit(1);
    }
});
