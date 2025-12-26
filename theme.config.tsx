import React from 'react'
import { DocsThemeConfig, useConfig } from 'nextra-theme-docs'
import { useRouter } from 'next/router'
import Link from 'next/link'

// Inline PostTags component
function PostTags({ tags }: { tags?: string[] }) {
    if (!tags || tags.length === 0) return null

    return (
        <div style={{
            display: 'flex',
            flexWrap: 'wrap',
            gap: '0.5rem',
            marginBottom: '1.5rem'
        }}>
            {tags.map(tag => (
                <Link
                    key={tag}
                    href={`/tags/${tag}`}
                    style={{
                        background: '#f0f0f0',
                        padding: '0.25rem 0.75rem',
                        borderRadius: '9999px',
                        fontSize: '0.85rem',
                        textDecoration: 'none',
                        color: '#555',
                    }}
                >
                    #{tag}
                </Link>
            ))}
        </div>
    )
}

const config: DocsThemeConfig = {
    logo: <span>Xiaozhi (Alan) Zhu</span>,
    project: {
        link: 'https://github.com/xiaozhi-alan-zhu/xiaozhi-alan-zhu.github.io',
    },
    //   chat: {
    //     link: 'https://discord.com',
    //   },
    docsRepositoryBase: 'https://github.com/xiaozhi-alan-zhu/xiaozhi-alan-zhu.github.io',
    footer: {
        text: '© 2024 Xiaozhi (Alan) Zhu',
    },
    useNextSeoProps: () => ({
        titleTemplate: '%s – Xiaozhi (Alan) Zhu'
    }),
    main: ({ children }) => {
        const { frontMatter } = useConfig()
        const router = useRouter()
        const isPost = router.pathname.startsWith('/posts/') && router.pathname !== '/posts'

        return (
            <div>
                {isPost && frontMatter?.tags && (
                    <PostTags tags={frontMatter.tags} />
                )}
                {children}
            </div>
        )
    }
}

export default config

