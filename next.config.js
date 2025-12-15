```javascript
module.exports = async () => {
    const remarkMath = (await import('remark-math')).default
    const rehypeKatex = (await import('rehype-katex')).default
    const rehypeCitation = (await import('rehype-citation')).default

    const withNextra = require('nextra')({
        theme: 'nextra-theme-docs',
        themeConfig: './theme.config.tsx',
        defaultShowCopyCode: true,
        mdxOptions: {
            remarkPlugins: [remarkMath],
            rehypePlugins: [
                [rehypeKatex, { 
                    trust: true, 
                    strict: false, 
                    output: 'html', // Use HTML output for better compatibility
                    macros: {
                        "\\eqref": "\\href{##1}{(\\text{#1})}",
                        "\\ref": "\\href{##1}{\\text{#1}}",
                        "\\label": "\\htmlId{#1}{}"
                    }
                }],
                [rehypeCitation, { bibliography: './public/papers.bib' }]
            ]
        }
    })

    return withNextra({
        output: 'export',
        images: {
            unoptimized: true
        }
    })
}
```
