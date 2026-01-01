
const fs = require('fs');
const path = require('path');
const matter = require('gray-matter');

const POSTS_DIR = path.join(process.cwd(), 'pages/posts');
const META_FILE = path.join(POSTS_DIR, '_meta.json');

// Base configuration
const meta = {
    "index": {
        "theme": {
            "sidebar": false,
            "toc": false,
            "breadcrumb": false
        }
    },
    "*": {
        "theme": {
            "sidebar": true,
            "toc": true,
            "breadcrumb": true
        }
    }
};

try {
    if (!fs.existsSync(POSTS_DIR)) {
        console.error(`Directory not found: ${POSTS_DIR}`);
        process.exit(0);
    }

    const filenames = fs.readdirSync(POSTS_DIR);
    let hiddenCount = 0;

    filenames.forEach(filename => {
        if (!filename.endsWith('.md') && !filename.endsWith('.mdx')) return;
        if (filename === 'index.mdx') return;

        try {
            const filePath = path.join(POSTS_DIR, filename);
            const fileContents = fs.readFileSync(filePath, 'utf8');
            const { data } = matter(fileContents);

            if (data.draft) {
                const routeName = filename.replace(/\.mdx?$/, '');
                meta[routeName] = {
                    display: 'hidden'
                };
                hiddenCount++;
                console.log(`[Draft Hider] Hiding draft: ${routeName}`);
            }
        } catch (e) {
            console.error(`Error reading ${filename}:`, e.message);
        }
    });

    fs.writeFileSync(META_FILE, JSON.stringify(meta, null, 2));
    console.log(`[Draft Hider] Updated _meta.json. Hidden ${hiddenCount} drafts.`);

} catch (err) {
    console.error('[Draft Hider] Failed:', err);
    process.exit(1);
}
