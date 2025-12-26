import Link from 'next/link'

interface PostTagsProps {
    tags?: string[]
}

export function PostTags({ tags }: PostTagsProps) {
    if (!tags || tags.length === 0) return null

    return (
        <div className="post-tags" style={{
            display: 'flex',
            flexWrap: 'wrap',
            gap: '0.5rem',
            marginTop: '0.5rem',
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
                        transition: 'background 0.2s'
                    }}
                    onMouseOver={(e) => e.currentTarget.style.background = '#e0e0e0'}
                    onMouseOut={(e) => e.currentTarget.style.background = '#f0f0f0'}
                >
                    #{tag}
                </Link>
            ))}
        </div>
    )
}
