import Link from 'next/link'

export function PostList({ posts }) {
    if (!posts) return null
    // Sort posts by date (newest first)
    const sortedPosts = [...posts].sort((a, b) => {
        return new Date(b.frontMatter.date).getTime() - new Date(a.frontMatter.date).getTime()
    })

    return (
        <div className="post-list">
            {sortedPosts.map((post) => (
                <div key={post.route} className="post-item" style={{ marginBottom: '2rem' }}>
                    <h3 className="post-title" style={{ marginTop: '0.5rem', marginBottom: '0.5rem' }}>
                        <Link href={post.route} className="post-link">
                            {post.frontMatter.title}
                        </Link>
                    </h3>
                    <div className="post-meta" style={{ fontSize: '0.9rem', color: '#666' }}>
                        <span className="post-date">{post.frontMatter.date}</span>
                        {post.frontMatter.tags && (
                            <span className="post-tags" style={{ marginLeft: '1rem' }}>
                                {post.frontMatter.tags.map(tag => (
                                    <Link key={tag} href={`/tags/${tag}`} style={{ marginRight: '0.5rem', background: '#eee', padding: '0.2rem 0.5rem', borderRadius: '4px', textDecoration: 'none', color: 'inherit' }}>
                                        #{tag}
                                    </Link>
                                ))}
                            </span>
                        )}
                    </div>
                    {post.frontMatter.description && (
                        <p className="post-description" style={{ marginTop: '0.5rem' }}>
                            {post.frontMatter.description}
                        </p>
                    )}
                </div>
            ))}
        </div>
    )
}
