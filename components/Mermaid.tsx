import React, { useEffect, useRef } from 'react';
import mermaid from 'mermaid';

mermaid.initialize({
    startOnLoad: false,
    theme: 'default',
    securityLevel: 'loose',
});

interface MermaidProps {
    chart: string;
}

const Mermaid: React.FC<MermaidProps> = ({ chart }) => {
    const containerRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (containerRef.current) {
            mermaid.contentLoaded();
            const id = `mermaid-${Math.random().toString(36).substr(2, 9)}`;
            const render = async () => {
                try {
                    // Clear previous content
                    containerRef.current!.innerHTML = '';
                    const { svg } = await mermaid.render(id, chart);
                    if (containerRef.current) {
                        containerRef.current.innerHTML = svg;
                    }
                } catch (error) {
                    console.error("Mermaid rendering failed:", error);
                    if (containerRef.current) {
                        containerRef.current.innerHTML = "Error rendering chart";
                    }
                }
            };
            render();
        }
    }, [chart]);

    return <div className="mermaid" ref={containerRef} />;
};

export default Mermaid;
