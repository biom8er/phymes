import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
import elkLayouts from 'https://cdn.jsdelivr.net/npm/@mermaid-js/layout-elk@0/dist/mermaid-layout-elk.esm.min.mjs';
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