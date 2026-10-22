import { describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { axe } from 'jest-axe';
import { App } from './App';

function renderAt(path: string) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <App />
    </MemoryRouter>,
  );
}

describe('App shell', () => {
  it('renders semantic landmarks and a skip link', () => {
    renderAt('/');
    expect(screen.getByRole('banner')).toBeInTheDocument();
    expect(screen.getByRole('main')).toBeInTheDocument();
    expect(screen.getByRole('contentinfo')).toBeInTheDocument();
    expect(
      screen.getByRole('navigation', { name: /primary/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole('link', { name: /skip to main content/i }),
    ).toBeInTheDocument();
  });

  it('routes to each placeholder page', () => {
    renderAt('/results');
    expect(
      screen.getByRole('heading', { level: 1, name: /results/i }),
    ).toBeInTheDocument();
  });

  it('shows the not-found page for an unknown route', () => {
    renderAt('/does-not-exist');
    expect(
      screen.getByRole('heading', { level: 1, name: /page not found/i }),
    ).toBeInTheDocument();
  });

  it('has no automatically-detectable accessibility violations on the shell', async () => {
    const { container } = renderAt('/');
    const results = await axe(container);
    expect(results).toHaveNoViolations();
  });
});
