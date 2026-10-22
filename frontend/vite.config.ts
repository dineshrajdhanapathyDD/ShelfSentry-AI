/// <reference types="vitest/config" />
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Vite + Vitest configuration for the ShelfSentry frontend (task 13.1).
//
// The test runner is configured for a single run (no watch): `npm test` maps to
// `vitest run`. The jsdom environment plus the global setup file wire React
// Testing Library + jest-dom matchers and the jest-axe accessibility matcher so
// later tasks (13.2-13.6) can assert on the four pages without extra setup.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
  },
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    css: true,
    // Single-run by default; watch mode is opt-in via `npm run test:watch`.
    watch: false,
    coverage: {
      provider: 'v8',
      reportsDirectory: './coverage',
      reporter: ['text', 'html'],
    },
  },
});
