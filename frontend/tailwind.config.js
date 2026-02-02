/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            fontFamily: {
                heading: ['Josefin Sans', 'sans-serif'],
                body: ['Inter', 'sans-serif'],
                mono: ['JetBrains Mono', 'monospace'],
            },
            colors: {
                verger: {
                    emerald: '#059669',
                    gold: '#d97706',
                    slate: '#0f172a',
                }
            }
        },
    },
    plugins: [],
}
