import React from 'react'
import { DocsThemeConfig } from 'nextra-theme-docs'

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
}

export default config
