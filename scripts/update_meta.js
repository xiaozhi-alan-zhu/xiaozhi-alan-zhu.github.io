
const fs = require('fs');
const path = require('path');
const matter = require('gray-matter');

const POSTS_DIR = path.join(process.cwd(), 'pages/posts');
const META_FILE = path.join(POSTS_DIR, '_meta.json');

try {
    if (!fs.existsSync(POSTS_DIR)) {
        console.error(`Directory not found: ${POSTS_DIR}`);
        process.exit(0);
    }

    const filenames = fs.readdirSync(POSTS_DIR);
    let hiddenCount = 0;

    // separate object for hidden items so we can merge them effectively
    const hiddenMeta = {};

    filenames.forEach(filename => {
        if (!filename.endsWith('.md') && !filename.endsWith('.mdx')) return;
        if (filename === 'index.mdx') return;

        try {
            const filePath = path.join(POSTS_DIR, filename);
            const fileContents = fs.readFileSync(filePath, 'utf8');
            const { data } = matter(fileContents);

            if (data.draft) {
                const routeName = filename.replace(/\.mdx?$/, '');
                hiddenMeta[routeName] = {
                    display: 'hidden'
                };
                hiddenCount++;
                console.log(`[Draft Hider] Hiding draft: ${routeName}`);
            }
        } catch (e) {
            console.error(`Error reading ${filename}:`, e.message);
        }
    });

    // Construct the final meta object.
    // Order matters in Nextra _meta.json sometimes. 
    // We place the specific hidden items BEFORE the wildcard to be safe, 
    // though explicit keys usually override wildcards anyway.
    const meta = {
        "index": {
            "theme": {
                "sidebar": false,
                "toc": false,
                "breadcrumb": false
            }
        },
        ...hiddenMeta, // Spread hidden items here
        "*": {
            "theme": {
                "sidebar": true,
                "toc": true,
                "breadcrumb": true
            }
        }
    };

    fs.writeFileSync(META_FILE, JSON.stringify(meta, null, 2));
    console.log(`[Draft Hider] Updated _meta.json. Hidden ${hiddenCount} drafts.`);

} catch (err) {
    console.error('[Draft Hider] Failed:', err);
    process.exit(1);
}
