/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ["var(--font-inter)"],
        mono: ["var(--font-ibm-plex-mono)"],
      },
      colors: {
        primary: {
          DEFAULT: "#1a1a1a", // Deep charcoal
          foreground: "#f8fafc", // Primary text
        },
        secondary: {
          DEFAULT: "#242424", // Lighter panels
          foreground: "#94a3b8", // Secondary text
        },
        muted: {
          DEFAULT: "#374151",
          foreground: "#64748b",
        },
        accent: {
          blue: "#00d4ff",
          green: "#00ff88",
          red: "#ff6b6b",
          yellow: "#ffcc02",
          purple: "#8b5cf6",
        },
      },
    },
  },
  plugins: [],
}; 