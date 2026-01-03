import React, { useEffect, useRef } from 'react';

const DinoArchitectureDiagram = () => {
    const containerRef = useRef<HTMLDivElement>(null);
    const svgRef = useRef<SVGSVGElement>(null);
    const animationRef = useRef<number | null>(null);

    // --- Styles ---
    // We embed the critical styles here to ensure self-containment without external Tailwind dependency
    const styles = {
        container: {
            position: 'relative',
            width: '100%',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            fontFamily: '"Google Sans", "Open Sans", sans-serif',
            padding: '1rem',
            backgroundColor: 'var(--surface-color, #ffffff)',
            color: 'var(--on-surface, #1B1C1D)',
            borderRadius: '12px',
            border: '1px solid var(--outline-variant, #C4C7C5)',
        } as React.CSSProperties,

        header: {
            width: '100%',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            marginBottom: '1rem',
            padding: '0 0.5rem',
        } as React.CSSProperties,

        legend: {
            display: 'flex',
            gap: '1rem',
            fontSize: '0.875rem',
        } as React.CSSProperties,

        legendItem: {
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
        } as React.CSSProperties,

        diagramArea: {
            position: 'relative',
            width: '100%',
            aspectRatio: '16/9',
            backgroundColor: 'var(--surface-color, #ffffff)',
            borderRadius: '12px',
            overflow: 'hidden',
            border: '1px solid var(--outline-variant, #eaeaea)',
        } as React.CSSProperties,

        nodeCard: (top: string, left: string, width: string, height: string, borderColor: string, bgColor: string) => ({
            position: 'absolute',
            top,
            left,
            width,
            height,
            border: `2px solid ${borderColor}`,
            backgroundColor: bgColor,
            borderRadius: '8px',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            textAlign: 'center',
            zIndex: 10,
            backdropFilter: 'blur(8px)',
            boxShadow: '0 2px 4px rgba(0,0,0,0.05)',
            transition: 'transform 0.2s',
            cursor: 'default',
        } as React.CSSProperties),

        label: {
            fontSize: '0.75rem',
            fontWeight: 'bold',
            marginTop: '4px',
        } as React.CSSProperties,

        subLabel: {
            fontSize: '0.65rem',
            opacity: 0.8,
        } as React.CSSProperties,

        pill: (bg: string) => ({
            marginTop: '6px',
            fontSize: '0.6rem',
            backgroundColor: bg,
            color: 'white',
            padding: '2px 6px',
            borderRadius: '999px',
        } as React.CSSProperties),
    };

    // --- Animation Logic ---
    useEffect(() => {
        if (!svgRef.current || !containerRef.current) return;

        const svg = svgRef.current;

        // Helper to get definition vars
        // We mock these for now to default Google colors if CSS vars aren't set
        const colors = {
            blue: '#1A73E8',
            tangerine: '#FA903E',
            red: '#D93025',
            outline: '#C4C7C5',
            text: '#575B5F'
        };

        const getElementCenter = (id: string) => {
            const el = containerRef.current?.querySelector(`#${id}`);
            if (!el || !containerRef.current) return { x: 0, y: 0, width: 0, height: 0, left: 0, right: 0 };

            const rect = el.getBoundingClientRect();
            // We need coordinates relative to the diagram-area (which is the parent of SVG)
            // The svg is absolute inset-0 in diagram-area
            const containerRect = svg.getBoundingClientRect();

            return {
                x: rect.left - containerRect.left + rect.width / 2,
                y: rect.top - containerRect.top + rect.height / 2,
                width: rect.width,
                height: rect.height,
                left: rect.left - containerRect.left,
                right: rect.left - containerRect.left + rect.width,
                top: rect.top - containerRect.top,
                bottom: rect.top - containerRect.top + rect.height
            };
        };

        const createPath = (id: string, startId: string, endId: string, type: 'solid' | 'ema', color: string | null) => {
            const start = getElementCenter(startId);
            const end = getElementCenter(endId);
            const existingPath = svg.getElementById(id);

            let d = '';
            if (type === 'ema') {
                // Vertical dashed line
                d = `M ${start.x} ${start.bottom} L ${end.x} ${end.top}`;
            } else if (startId === 'node-input') {
                // Split input
                const startX = start.right;
                const startY = start.y;
                const endX = end.left;
                const endY = end.y;
                const midX = (startX + endX) / 2;
                d = `M ${startX} ${startY} C ${midX} ${startY}, ${midX} ${endY}, ${endX} ${endY}`;
            } else {
                const startX = start.right;
                const startY = start.y;
                const endX = end.left;
                const endY = end.y;

                if (endId === 'node-koleo') {
                    d = `M ${startX} ${startY} C ${startX + 40} ${startY}, ${endX - 40} ${endY}, ${endX} ${endY}`;
                } else if (endId === 'node-ibot') {
                    d = `M ${startX} ${startY} C ${startX + 40} ${startY}, ${endX - 40} ${endY}, ${endX} ${endY}`;
                } else {
                    d = `M ${startX} ${startY} L ${endX} ${endY}`;
                }
            }

            let path: SVGPathElement;
            if (existingPath) {
                path = existingPath as unknown as SVGPathElement;
            } else {
                path = document.createElementNS("http://www.w3.org/2000/svg", "path");
                path.setAttribute("id", id);
                path.setAttribute("class", "connection-line");
                svg.appendChild(path);
            }

            path.setAttribute("d", d);
            path.style.fill = 'none';
            path.style.stroke = color || colors.outline;
            path.style.strokeWidth = '2';
            path.style.strokeLinecap = 'round';

            if (type === 'ema') {
                path.style.strokeDasharray = '6, 6';
            } else {
                path.setAttribute("marker-end", "url(#arrowhead)");
            }

            return path;
        };

        const drawPaths = () => {
            // Define connections
            createPath('p1', 'node-input', 'node-student', 'solid', null);
            createPath('p2', 'node-input', 'node-teacher', 'solid', null);
            createPath('p3', 'node-student', 'node-koleo', 'solid', null);
            createPath('p4', 'node-student', 'node-dino', 'solid', null);
            createPath('p5', 'node-teacher', 'node-dino', 'solid', null);
            createPath('p6', 'node-student', 'node-ibot', 'solid', null);
            createPath('p7', 'node-teacher', 'node-ibot', 'solid', null);
            createPath('p8', 'node-student', 'node-teacher', 'ema', colors.text);
        };

        // Animation Loop
        let frame = 0;
        const particles: HTMLElement[] = [];

        const spawnParticle = (pathId: string, color: string, speed: number) => {
            const path = svg.querySelector(`#${pathId}`) as SVGPathElement;
            if (!path) return;

            const particle = document.createElement('div');
            particle.style.position = 'absolute';
            particle.style.width = '8px';
            particle.style.height = '8px';
            particle.style.borderRadius = '50%';
            particle.style.backgroundColor = color;
            particle.style.pointerEvents = 'none';
            particle.style.zIndex = '5';

            // Append to the diagram area, not SVG, so we can use top/left easily
            containerRef.current?.querySelector('#diagram-area')?.appendChild(particle);
            particles.push(particle);

            const length = path.getTotalLength();
            let progress = 0;

            const move = () => {
                if (!containerRef.current) return; // Stop if unmounted
                progress += speed;
                if (progress >= length) {
                    particle.remove();
                    const idx = particles.indexOf(particle);
                    if (idx > -1) particles.splice(idx, 1);
                    return;
                }

                const point = path.getPointAtLength(progress);
                // Adjust for diagram area relative coords
                // NOTE: getPointAtLength returns coordinates in SVG user space
                // Since SVG is absolute inset-0, these match the parent div's coords
                const svgRect = svg.getBoundingClientRect();

                particle.style.left = `${point.x - 4}px`;
                particle.style.top = `${point.y - 4}px`;

                requestAnimationFrame(move);
            };
            requestAnimationFrame(move);
        };

        const animate = () => {
            frame++;

            // Spawn inputs
            if (frame % 100 === 0) {
                spawnParticle('p1', colors.blue, 2);
                spawnParticle('p2', colors.blue, 2);
            }

            // Spawn secondary flows (delayed)
            if (frame % 100 === 50) {
                spawnParticle('p3', colors.blue, 2);
                spawnParticle('p4', colors.blue, 2);
                spawnParticle('p6', colors.blue, 2);
                spawnParticle('p8', colors.text, 1); // EMA
            }

            if (frame % 100 === 50) {
                spawnParticle('p5', colors.tangerine, 2);
                spawnParticle('p7', colors.tangerine, 2);
            }

            animationRef.current = requestAnimationFrame(animate);
        };

        // Initialize
        drawPaths();
        animate();

        // Handle Resize
        const handleResize = () => {
            drawPaths();
        };
        window.addEventListener('resize', handleResize);

        return () => {
            window.removeEventListener('resize', handleResize);
            if (animationRef.current) cancelAnimationFrame(animationRef.current);
            particles.forEach(p => p.remove());
        };
    }, []);

    return (
        <div ref={containerRef} style={styles.container}>
            {/* Defines */}
            <svg style={{ position: 'absolute', width: 0, height: 0 }}>
                <defs>
                    <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                        <polygon points="0 0, 10 3.5, 0 7" fill="#C4C7C5" />
                    </marker>
                </defs>
            </svg>

            <div style={styles.header}>
                <h4 style={{ margin: 0, fontSize: '1rem', color: '#575B5F' }}>系统数据流向图</h4>
                <div style={styles.legend}>
                    <div style={styles.legendItem}><div style={{ width: 10, height: 10, borderRadius: '50%', background: '#1A73E8' }} /> Student</div>
                    <div style={styles.legendItem}><div style={{ width: 10, height: 10, borderRadius: '50%', background: '#FA903E' }} /> Teacher</div>
                    <div style={styles.legendItem}><div style={{ width: 10, height: 10, borderRadius: '50%', background: '#D93025' }} /> Loss</div>
                </div>
            </div>

            <div id="diagram-area" style={styles.diagramArea}>
                <svg ref={svgRef} id="connections" style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none' }} />

                {/* Node: Input */}
                <div id="node-input" style={styles.nodeCard('40%', '5%', '12%', '20%', '#C4C7C5', '#ffffff')}>
                    <div style={{ fontSize: '1.5rem' }}>🖼️</div>
                    <div style={styles.label}>Input</div>
                    <div style={styles.subLabel}>Crops</div>
                </div>

                {/* Node: Student */}
                <div id="node-student" style={styles.nodeCard('10%', '30%', '18%', '25%', '#1A73E8', '#e8f0fe')}>
                    <div style={{ ...styles.label, color: '#1A73E8' }}>Student</div>
                    <div style={{ ...styles.subLabel, color: '#1A73E8' }}>ViT Encoder</div>
                    <div style={styles.pill('#1A73E8')}>Backprop</div>
                </div>

                {/* Node: Teacher */}
                <div id="node-teacher" style={styles.nodeCard('65%', '30%', '18%', '25%', '#FA903E', '#fff3e0')}>
                    <div style={{ ...styles.label, color: '#FA903E' }}>Teacher</div>
                    <div style={{ ...styles.subLabel, color: '#FA903E' }}>ViT Encoder</div>
                    <div style={styles.pill('#FA903E')}>Stop Grad</div>
                </div>

                <div style={{ position: 'absolute', left: '55%', top: '10%', bottom: '10%', borderRight: '1px dashed #C4C7C5' }} />
                <div style={{ position: 'absolute', left: '56%', top: '5%', fontSize: '0.7rem', color: '#777', transform: 'rotate(90deg)', transformOrigin: 'left' }}>Projection Heads</div>

                {/* Node: KoLeo */}
                <div id="node-koleo" style={styles.nodeCard('5%', '79%', '16%', '18%', '#D93025', '#fce8e6')}>
                    <div style={{ ...styles.label, color: '#D93025' }}>KoLeo Reg</div>
                    <div style={styles.subLabel}>Entropy</div>
                </div>

                {/* Node: DINO */}
                <div id="node-dino" style={styles.nodeCard('40%', '79%', '16%', '20%', '#D93025', '#fce8e6')}>
                    <div style={{ ...styles.label, color: '#D93025' }}>DINO Loss</div>
                    <div style={styles.subLabel}>Global</div>
                </div>

                {/* Node: iBOT */}
                <div id="node-ibot" style={styles.nodeCard('75%', '79%', '16%', '20%', '#D93025', '#fce8e6')}>
                    <div style={{ ...styles.label, color: '#D93025' }}>iBOT Loss</div>
                    <div style={styles.subLabel}>Local/MIM</div>
                </div>

                <div style={{ position: 'absolute', left: '39%', top: '50%', transform: 'translate(-50%, -50%)', fontSize: '0.6rem', background: 'white', padding: '2px 4px', border: '1px solid #ccc', borderRadius: '4px', zIndex: 20 }}>
                    EMA
                </div>

            </div>
        </div>
    );
};

export default DinoArchitectureDiagram;
