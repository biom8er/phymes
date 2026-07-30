import mermaid from './mermaid.esm.min.mjs';
import elkLayouts from './mermaid-layout-elk.esm.min.mjs';
window.mermaid = mermaid;
mermaid.registerLayoutLoaders(elkLayouts);
mermaid.initialize({
    theme: "dark",
    startOnLoad: false,
    maxTextSize: 50000,
    maxEdges: 500,
    securityLevel: "loose",
    suppressErrorRendering: true
})