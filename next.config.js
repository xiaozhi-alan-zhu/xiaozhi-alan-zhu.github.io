
const withNextra = require('nextra')({
    theme: 'nextra-theme-docs',
    themeConfig: './theme.config.tsx',
    latex: true,
    mdxOptions: {
        rehypePlugins: [
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
