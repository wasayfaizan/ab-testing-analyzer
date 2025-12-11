/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        // Primary gradient: Electric Violet → Royal Purple
        "electric-violet": "#8A3FFC",
        "royal-purple": "#7B2FFF",
        // Secondary accent: Aqua / Teal
        aqua: "#25E6D1",
        // Background colors
        "dark-navy": "#0D0F14", // Page background
        "card-bg": "#13161C", // Card background (lighter than page)
        // Text colors
        "text-primary": "#FFFFFF",
        "text-secondary": "#A8A8B3",
        // Badge colors
        "badge-positive": "#25E6D1",
        "badge-significant": "#FBBF24", // Gold
        "badge-neutral": "#94A3B8", // Slate grey
      },
      backgroundImage: {
        "gradient-primary": "linear-gradient(135deg, #8A3FFC 0%, #7B2FFF 100%)",
        "gradient-card": "linear-gradient(135deg, #0D0F14 0%, #151821 100%)",
      },
      animation: {
        "fade-in": "fade-in 0.5s ease-out",
        "slide-down": "slide-down 0.3s ease-out",
      },
    },
  },
  plugins: [require("@tailwindcss/typography")],
};
