// Tailwind CSS v4 is wired through its dedicated PostCSS plugin.
// Autoprefixer is kept for vendor-prefix coverage across target browsers.
export default {
  plugins: {
    '@tailwindcss/postcss': {},
    autoprefixer: {},
  },
};
