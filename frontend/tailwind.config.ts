import type { Config } from 'tailwindcss'

const config: Config = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        // Bloomberg-inspired color palette
        bloomberg: {
          black: '#0f0f23',
          dark: '#1a1b3a',
          darker: '#13142a',
          orange: '#ff6b1a',
          'orange-light': '#ff8542',
          'orange-dark': '#e55100',
          amber: '#ffab00',
          yellow: '#ffd600',
          green: '#00e676',
          'green-dark': '#00c853',
          red: '#ff1744',
          'red-dark': '#d50000',
          blue: '#2196f3',
          cyan: '#00e5ff',
          purple: '#7c4dff',
          pink: '#e91e63',
          gray: {
            100: '#f5f5f5',
            200: '#eeeeee',
            300: '#e0e0e0',
            400: '#bdbdbd',
            500: '#9e9e9e',
            600: '#757575',
            700: '#616161',
            800: '#424242',
            900: '#212121',
          }
        },
        surface: {
          primary: '#1a1b3a',
          secondary: '#252659',
          tertiary: '#2d3071',
          elevated: '#373a87',
          hover: '#404399',
        },
        border: {
          primary: '#404399',
          secondary: '#2d3071',
          accent: '#ff6b1a',
        },
        text: {
          primary: '#ffffff',
          secondary: '#b0bec5',
          muted: '#78909c',
          success: '#00e676',
          danger: '#ff1744',
          warning: '#ffab00',
          info: '#00e5ff',
        }
      },
      fontFamily: {
        sans: ['Inter', 'Segoe UI', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'SF Mono', 'monospace'],
      },
      fontSize: {
        'xs': ['0.75rem', { lineHeight: '1rem' }],
        'sm': ['0.875rem', { lineHeight: '1.25rem' }],
        'base': ['1rem', { lineHeight: '1.5rem' }],
        'lg': ['1.125rem', { lineHeight: '1.75rem' }],
        'xl': ['1.25rem', { lineHeight: '1.75rem' }],
        '2xl': ['1.5rem', { lineHeight: '2rem' }],
        '3xl': ['1.875rem', { lineHeight: '2.25rem' }],
        '4xl': ['2.25rem', { lineHeight: '2.5rem' }],
        '5xl': ['3rem', { lineHeight: '1' }],
        '6xl': ['3.75rem', { lineHeight: '1' }],
      },
      spacing: {
        '18': '4.5rem',
        '72': '18rem',
        '84': '21rem',
        '96': '24rem',
      },
      borderRadius: {
        'none': '0',
        'sm': '0.125rem',
        DEFAULT: '0.25rem',
        'md': '0.375rem',
        'lg': '0.5rem',
        'xl': '0.75rem',
        '2xl': '1rem',
        '3xl': '1.5rem',
        'full': '9999px',
      },
      boxShadow: {
        'sm': '0 1px 2px 0 rgba(15, 15, 35, 0.05)',
        DEFAULT: '0 1px 3px 0 rgba(15, 15, 35, 0.1), 0 1px 2px 0 rgba(15, 15, 35, 0.06)',
        'md': '0 4px 6px -1px rgba(15, 15, 35, 0.1), 0 2px 4px -1px rgba(15, 15, 35, 0.06)',
        'lg': '0 10px 15px -3px rgba(15, 15, 35, 0.1), 0 4px 6px -2px rgba(15, 15, 35, 0.05)',
        'xl': '0 20px 25px -5px rgba(15, 15, 35, 0.1), 0 10px 10px -5px rgba(15, 15, 35, 0.04)',
        '2xl': '0 25px 50px -12px rgba(15, 15, 35, 0.25)',
        'inner': 'inset 0 2px 4px 0 rgba(15, 15, 35, 0.06)',
        'glow': '0 0 20px rgba(255, 107, 26, 0.5)',
        'glow-sm': '0 0 10px rgba(255, 107, 26, 0.3)',
        'glow-lg': '0 0 40px rgba(255, 107, 26, 0.6)',
        'terminal': '0 8px 32px rgba(15, 15, 35, 0.8)',
      },
      animation: {
        'pulse-orange': 'pulse-orange 2s infinite',
        'slide-up': 'slide-up 0.3s ease-out',
        'fade-in': 'fade-in 0.5s ease-in-out',
        'bounce-subtle': 'bounce-subtle 1s infinite',
        'glow': 'glow 2s ease-in-out infinite alternate',
      },
      keyframes: {
        'pulse-orange': {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.7' },
        },
        'slide-up': {
          from: { transform: 'translateY(20px)', opacity: '0' },
          to: { transform: 'translateY(0)', opacity: '1' },
        },
        'fade-in': {
          from: { opacity: '0' },
          to: { opacity: '1' },
        },
        'bounce-subtle': {
          '0%, 100%': { transform: 'translateY(-5%)' },
          '50%': { transform: 'translateY(0)' },
        },
        'glow': {
          from: { boxShadow: '0 0 20px rgba(255, 107, 26, 0.5)' },
          to: { boxShadow: '0 0 30px rgba(255, 107, 26, 0.8)' },
        },
      },
      backdropBlur: {
        xs: '2px',
      },
      backgroundImage: {
        'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
        'gradient-conic': 'conic-gradient(from 180deg at 50% 50%, var(--tw-gradient-stops))',
        'terminal-bg': 'linear-gradient(135deg, #0f0f23 0%, #1a1b3a 100%)',
        'header-bg': 'linear-gradient(90deg, #1a1b3a 0%, #252659 100%)',
      },
      screens: {
        'xs': '475px',
        '3xl': '1920px',
      },
      gridTemplateColumns: {
        'terminal': 'minmax(250px, 1fr) 3fr minmax(250px, 1fr)',
        'dashboard': 'repeat(auto-fit, minmax(300px, 1fr))',
      },
      zIndex: {
        '60': '60',
        '70': '70',
        '80': '80',
        '90': '90',
        '100': '100',
      },
    },
  },
  plugins: [
    // Custom plugin for Bloomberg-specific utilities
    function({ addUtilities, theme, addComponents }) {
      const newUtilities = {
        '.text-glow': {
          textShadow: '0 0 8px currentColor',
        },
        '.border-glow': {
          boxShadow: '0 0 8px #ff6b1a',
        },
        '.glass-effect': {
          backdropFilter: 'blur(10px)',
          background: 'rgba(26, 27, 58, 0.8)',
          border: '1px solid rgba(255, 107, 26, 0.2)',
        },
        '.terminal-scrollbar': {
          '&::-webkit-scrollbar': {
            width: '8px',
            height: '8px',
          },
          '&::-webkit-scrollbar-track': {
            background: '#13142a',
            borderRadius: '4px',
          },
          '&::-webkit-scrollbar-thumb': {
            background: '#ff6b1a',
            borderRadius: '4px',
            transition: 'background 0.2s ease',
          },
          '&::-webkit-scrollbar-thumb:hover': {
            background: '#ff8542',
          },
        },
      }

      const newComponents = {
        '.btn': {
          display: 'inline-flex',
          alignItems: 'center',
          justifyContent: 'center',
          padding: '0.5rem 1rem',
          border: 'none',
          borderRadius: '0.375rem',
          fontWeight: '500',
          fontSize: '0.875rem',
          cursor: 'pointer',
          transition: 'all 0.2s ease',
          textDecoration: 'none',
          fontFamily: 'inherit',
        },
        '.btn-primary': {
          background: '#ff6b1a',
          color: 'white',
          '&:hover': {
            background: '#e55100',
            transform: 'translateY(-1px)',
            boxShadow: '0 4px 12px rgba(255, 107, 26, 0.3)',
          },
        },
        '.btn-secondary': {
          background: '#252659',
          color: '#ffffff',
          border: '1px solid #404399',
          '&:hover': {
            background: '#2d3071',
            borderColor: '#ff6b1a',
          },
        },
        '.terminal-input': {
          background: '#1a1b3a',
          border: '1px solid #2d3071',
          borderRadius: '0.375rem',
          padding: '0.5rem 0.75rem',
          color: '#ffffff',
          fontSize: '0.875rem',
          transition: 'all 0.2s ease',
          fontFamily: theme('fontFamily.mono').join(', '),
          '&:focus': {
            outline: 'none',
            borderColor: '#ff6b1a',
            boxShadow: '0 0 0 2px rgba(255, 107, 26, 0.2)',
          },
          '&::placeholder': {
            color: '#78909c',
          },
        },
        '.terminal-card': {
          background: '#1a1b3a',
          border: '1px solid #2d3071',
          borderRadius: '0.5rem',
          overflow: 'hidden',
          transition: 'all 0.2s ease',
          '&:hover': {
            borderColor: '#ff6b1a',
            boxShadow: '0 4px 20px rgba(15, 15, 35, 0.5)',
          },
        },
        '.price-positive': {
          color: '#00e676',
        },
        '.price-negative': {
          color: '#ff1744',
        },
        '.price-neutral': {
          color: '#b0bec5',
        },
      }

      addUtilities(newUtilities)
      addComponents(newComponents)
    },
  ],
}

export default config