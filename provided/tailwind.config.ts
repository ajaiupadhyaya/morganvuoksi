import type { Config } from "tailwindcss";

export default {
	darkMode: ["class"],
	content: [
		"./pages/**/*.{ts,tsx}",
		"./components/**/*.{ts,tsx}",
		"./app/**/*.{ts,tsx}",
		"./src/**/*.{ts,tsx}",
	],
	prefix: "",
	theme: {
		container: {
			center: true,
			padding: '2rem',
			screens: {
				'2xl': '1400px'
			}
		},
		extend: {
			colors: {
				border: 'hsl(var(--border))',
				input: 'hsl(var(--input))',
				ring: 'hsl(var(--ring))',
				background: 'hsl(var(--background))',
				foreground: 'hsl(var(--foreground))',
				primary: {
					DEFAULT: 'hsl(var(--primary))',
					foreground: 'hsl(var(--primary-foreground))'
				},
				secondary: {
					DEFAULT: 'hsl(var(--secondary))',
					foreground: 'hsl(var(--secondary-foreground))'
				},
				destructive: {
					DEFAULT: 'hsl(var(--destructive))',
					foreground: 'hsl(var(--destructive-foreground))'
				},
				muted: {
					DEFAULT: 'hsl(var(--muted))',
					foreground: 'hsl(var(--muted-foreground))'
				},
				accent: {
					DEFAULT: 'hsl(var(--accent))',
					foreground: 'hsl(var(--accent-foreground))'
				},
				popover: {
					DEFAULT: 'hsl(var(--popover))',
					foreground: 'hsl(var(--popover-foreground))'
				},
				card: {
					DEFAULT: 'hsl(var(--card))',
					foreground: 'hsl(var(--card-foreground))'
				},
				sidebar: {
					DEFAULT: 'hsl(var(--sidebar-background))',
					foreground: 'hsl(var(--sidebar-foreground))',
					primary: 'hsl(var(--sidebar-primary))',
					'primary-foreground': 'hsl(var(--sidebar-primary-foreground))',
					accent: 'hsl(var(--sidebar-accent))',
					'accent-foreground': 'hsl(var(--sidebar-accent-foreground))',
					border: 'hsl(var(--sidebar-border))',
					ring: 'hsl(var(--sidebar-ring))'
				},
				// PROFESSIONAL BLOOMBERG TERMINAL COLOR SYSTEM
				terminal: {
					// Core Terminal Colors - Exact Professional Specification
					bg: '#000000',           // Deep black background - exact Bloomberg match
					panel: '#0a0a0a',        // Slightly lighter panels for contrast
					border: '#333333',       // Professional gray borders
					text: '#ffffff',         // Pure white text for maximum contrast
					muted: '#888888',        // Muted text for secondary information
					
					// Bloomberg Terminal Signature Colors
					orange: '#ff6b35',       // Bloomberg orange - primary accent
					amber: '#ffa500',        // Amber for warnings and highlights
					cyan: '#00d4ff',         // Bright cyan - primary data color
					blue: '#0088cc',         // Professional blue for headers
					
					// Financial Data Colors
					green: '#00ff88',        // Bullish/positive values - bright green
					red: '#ff4757',          // Bearish/negative values - bright red
					
					// Additional Professional Colors
					purple: '#8b5cf6',       // Analysis highlights
					yellow: '#ffd700',       // Important alerts
					
					// Neutral Grays for Data Hierarchy
					gray: {
						100: '#f7f7f7',
						200: '#e5e5e5',
						300: '#d4d4d4',
						400: '#a3a3a3',
						500: '#737373',
						600: '#525252',
						700: '#404040',
						800: '#262626',
						900: '#171717',
					}
				}
			},
			borderRadius: {
				lg: 'var(--radius)',
				md: 'calc(var(--radius) - 2px)',
				sm: 'calc(var(--radius) - 4px)'
			},
			// Professional Terminal Typography
			fontFamily: {
				mono: [
					'JetBrains Mono', 
					'Monaco', 
					'Consolas', 
					'Courier New', 
					'SFMono-Regular', 
					'ui-monospace', 
					'monospace'
				],
				sans: [
					'Inter', 
					'system-ui', 
					'-apple-system', 
					'sans-serif'
				],
			},
			// Terminal-specific spacing for ultra-dense layouts
			spacing: {
				'0.25': '0.0625rem',  // 1px
				'18': '4.5rem',       // 72px
				'22': '5.5rem',       // 88px
				'26': '6.5rem',       // 104px
				'30': '7.5rem',       // 120px
			},
			// Professional grid system for Bloomberg layouts
			gridTemplateColumns: {
				'16': 'repeat(16, minmax(0, 1fr))',
				'20': 'repeat(20, minmax(0, 1fr))',
				'24': 'repeat(24, minmax(0, 1fr))',
			},
			// Bloomberg Terminal specific animations
			keyframes: {
				'accordion-down': {
					from: {
						height: '0'
					},
					to: {
						height: 'var(--radix-accordion-content-height)'
					}
				},
				'accordion-up': {
					from: {
						height: 'var(--radix-accordion-content-height)'
					},
					to: {
						height: '0'
					}
				},
				// Terminal-specific animations
				'terminal-pulse': {
					'0%, 100%': {
						opacity: '1'
					},
					'50%': {
						opacity: '0.7'
					}
				},
				'data-flash': {
					'0%': { 
						backgroundColor: 'transparent' 
					},
					'50%': { 
						backgroundColor: '#00d4ff',
						color: '#000000'
					},
					'100%': { 
						backgroundColor: 'transparent' 
					}
				},
				'ticker-scroll': {
					'0%': { 
						transform: 'translateX(100%)' 
					},
					'100%': { 
						transform: 'translateX(-100%)' 
					}
				},
				'chart-draw': {
					'0%': { 
						strokeDashoffset: '1000' 
					},
					'100%': { 
						strokeDashoffset: '0' 
					}
				},
				'slide-up-terminal': {
					'0%': {
						transform: 'translateY(10px)',
						opacity: '0'
					},
					'100%': {
						transform: 'translateY(0)',
						opacity: '1'
					}
				},
				'glow-pulse': {
					'0%, 100%': {
						boxShadow: '0 0 5px rgba(0, 212, 255, 0.5)'
					},
					'50%': {
						boxShadow: '0 0 20px rgba(0, 212, 255, 0.8), 0 0 30px rgba(0, 212, 255, 0.4)'
					}
				}
			},
			animation: {
				'accordion-down': 'accordion-down 0.2s ease-out',
				'accordion-up': 'accordion-up 0.2s ease-out',
				'terminal-pulse': 'terminal-pulse 2s ease-in-out infinite',
				'data-flash': 'data-flash 0.3s ease-in-out',
				'ticker-scroll': 'ticker-scroll 30s linear infinite',
				'chart-draw': 'chart-draw 2s ease-in-out',
				'slide-up-terminal': 'slide-up-terminal 0.3s ease-out',
				'glow-pulse': 'glow-pulse 2s ease-in-out infinite'
			},
			// Professional box shadows for terminal depth
			boxShadow: {
				'terminal': '0 2px 4px rgba(0, 0, 0, 0.8), inset 0 1px 0 rgba(0, 212, 255, 0.1)',
				'terminal-inset': 'inset 0 2px 4px rgba(0, 0, 0, 0.8)',
				'terminal-glow': '0 0 20px rgba(0, 212, 255, 0.3)',
				'terminal-orange-glow': '0 0 20px rgba(255, 107, 53, 0.4)',
				'terminal-green-glow': '0 0 20px rgba(0, 255, 136, 0.4)',
				'terminal-deep': '0 4px 8px rgba(0, 0, 0, 0.9), 0 2px 4px rgba(0, 0, 0, 0.8)',
			},
			// Professional backdrop filters
			backdropBlur: {
				'terminal': '8px',
			},
			// Letter spacing for terminal density
			letterSpacing: {
				'terminal': '0.025em',
				'terminal-wide': '0.1em',
				'terminal-wider': '0.15em',
			},
			// Line heights for information density
			lineHeight: {
				'terminal': '1.1',
				'terminal-tight': '1.0',
				'terminal-loose': '1.3',
			},
			// Professional screen breakpoints
			screens: {
				'terminal': '1920px',  // Professional terminal resolution
				'4k': '3840px',        // 4K trading screens
			}
		}
	},
	plugins: [
		require("tailwindcss-animate"),
		// Custom plugin for terminal-specific utilities
		function({ addUtilities }: { addUtilities: any }) {
			const newUtilities = {
				'.terminal-panel': {
					background: 'linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%)',
					border: '1px solid #333333',
					boxShadow: 'inset 0 1px 0 rgba(0, 212, 255, 0.1), 0 2px 4px rgba(0, 0, 0, 0.8)',
				},
				'.terminal-button': {
					background: 'linear-gradient(135deg, #1a1a1a 0%, #0a0a0a 100%)',
					border: '1px solid #333333',
					color: '#ffffff',
					fontSize: '0.75rem',
					fontFamily: 'JetBrains Mono, Monaco, Consolas, monospace',
					fontWeight: '500',
					textTransform: 'uppercase',
					letterSpacing: '0.05em',
					padding: '0.25rem 0.5rem',
					transition: 'all 0.15s ease',
				},
				'.terminal-button:hover': {
					borderColor: '#ff6b35',
					color: '#ff6b35',
					background: 'linear-gradient(135deg, #2a2a2a 0%, #1a1a1a 100%)',
					boxShadow: '0 0 10px rgba(255, 107, 53, 0.3)',
				},
				'.terminal-input': {
					background: 'linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%)',
					border: '1px solid #333333',
					color: '#ffffff',
					fontSize: '0.75rem',
					fontFamily: 'JetBrains Mono, Monaco, Consolas, monospace',
					padding: '0.25rem 0.5rem',
				},
				'.terminal-input:focus': {
					borderColor: '#00d4ff',
					outline: 'none',
					boxShadow: '0 0 0 1px #00d4ff, 0 0 10px rgba(0, 212, 255, 0.3)',
				},
				'.financial-number': {
					fontFamily: 'JetBrains Mono, Monaco, Consolas, monospace',
					fontVariantNumeric: 'tabular-nums',
					fontWeight: '600',
					letterSpacing: '0.1em',
				},
				'.status-positive': {
					color: '#00ff88',
					fontWeight: '700',
					textShadow: '0 0 8px rgba(0, 255, 136, 0.6)',
				},
				'.status-negative': {
					color: '#ff4757',
					fontWeight: '700',
					textShadow: '0 0 8px rgba(255, 71, 87, 0.6)',
				},
				'.status-neutral': {
					color: '#00d4ff',
					fontWeight: '700',
					textShadow: '0 0 8px rgba(0, 212, 255, 0.6)',
				},
				'.glow-orange': {
					boxShadow: '0 0 10px rgba(255, 107, 53, 0.4), 0 0 20px rgba(255, 107, 53, 0.2), 0 0 40px rgba(255, 107, 53, 0.1)',
				},
				'.glow-cyan': {
					boxShadow: '0 0 10px rgba(0, 212, 255, 0.4), 0 0 20px rgba(0, 212, 255, 0.2), 0 0 40px rgba(0, 212, 255, 0.1)',
				},
				'.glow-green': {
					boxShadow: '0 0 10px rgba(0, 255, 136, 0.4), 0 0 20px rgba(0, 255, 136, 0.2)',
				},
				'.dense-layout': {
					lineHeight: '1.1',
					letterSpacing: '0.02em',
				},
				'.ultra-dense': {
					lineHeight: '1.0',
					fontSize: '0.625rem',
					letterSpacing: '0.01em',
				},
				'.terminal-chart': {
					background: 'radial-gradient(circle at center, #1a1a1a 0%, #0a0a0a 100%)',
				},
				'.market-ticker': {
					background: 'linear-gradient(90deg, #1a1a1a 0%, #0a0a0a 50%, #1a1a1a 100%)',
					borderTop: '1px solid #333333',
					borderBottom: '1px solid #333333',
					overflow: 'hidden',
				},
				'.command-line': {
					background: 'linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%)',
					borderTop: '1px solid #333333',
				},
			}
			addUtilities(newUtilities)
		}
	],
} satisfies Config;
