import Giscus from '@giscus/react';
import { useTheme } from 'next-themes';
import { useEffect, useState } from 'react';

export default function Comments() {
    const { theme, systemTheme } = useTheme();
    const [mounted, setMounted] = useState(false);

    useEffect(() => {
        setMounted(true);
    }, []);

    if (!mounted) return null;

    return (
        <div style={{ marginTop: '50px', borderTop: '1px solid #eaeaea', paddingTop: '20px' }}>
            <h3>Comments</h3>
            <Giscus
                id="comments"
                repo="xiaozhi-alan-zhu/xiaozhi-alan-zhu.github.io"
                repoId="R_kgDOP1cUIg"
                category="General"
                categoryId="DIC_kwDOP1cUIs4Cv4rp"
                mapping="pathname"
                strict="0"
                reactionsEnabled="1"
                emitMetadata="0"
                inputPosition="bottom"
                theme={theme === "system" ? systemTheme : theme}
                lang="en"
                loading="lazy"
            />
        </div>
    );
}
