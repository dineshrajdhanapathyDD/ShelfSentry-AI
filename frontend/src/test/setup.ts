/**
 * Global test setup for Vitest (task 13.1).
 *
 * Wires:
 *  - `@testing-library/jest-dom` matchers (e.g. `toBeInTheDocument`).
 *  - `jest-axe`'s `toHaveNoViolations` accessibility matcher, registered on
 *    Vitest's `expect` so later tasks (13.6) can assert WCAG conformance.
 *  - automatic DOM cleanup between tests.
 *
 * The runner is single-run (no watch) via `vitest run` / `watch: false` in the
 * Vite config.
 */
import '@testing-library/jest-dom/vitest';
import { afterEach, expect } from 'vitest';
import { cleanup } from '@testing-library/react';
import { toHaveNoViolations } from 'jest-axe';

expect.extend(toHaveNoViolations);

afterEach(() => {
  cleanup();
});
