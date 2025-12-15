
const withNextra = require('nextra')({
    theme: 'nextra-theme-docs',
    themeConfig: './theme.config.tsx',
    defaultShowCopyCode: true,
    mdxOptions: {
        remarkPlugins: [require('remark-math')],
        rehypePlugins: [
            [require('rehype-katex'), {
                trust: true,
                strict: false,
                output: 'html', // Use HTML output for better compatibility
                macros: {
                    "\\eqref": "\\href{##1}{(\\text{#1})}",
                    "\\ref": "\\href{##1}{\\text{#1}}",
                    "\\label": "\\htmlId{#1}{}"
                }
            }],
            [require('rehype-citation'), { bibliography: './public/papers.bib' }]
        ]
    }
})

module.exports = withNextra({
    output: 'export',
    images: {
        unoptimized: true
    }
})
