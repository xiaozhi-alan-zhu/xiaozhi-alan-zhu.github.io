import 'katex/dist/katex.min.css'
import '../styles/globals.css'
import Script from 'next/script'

export default function App({ Component, pageProps }) {
    return (
        <>
            <Script
                src="https://www.googletagmanager.com/gtag/js?id=G-L37FLSK1TJ"
                strategy="afterInteractive"
            />
            <Script id="google-analytics" strategy="afterInteractive">
                {`
          window.dataLayer = window.dataLayer || [];
          function gtag(){window.dataLayer.push(arguments);}
          gtag('js', new Date());

          gtag('config', 'G-L37FLSK1TJ');
        `}
            </Script>
            <Component {...pageProps} />
        </>
    )
}
