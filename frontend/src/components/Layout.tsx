import { NavLink, Outlet } from 'react-router-dom';

/**
 * The minimal accessible app shell (task 13.1).
 *
 * Provides semantic landmarks (`<header>`, `<nav>`, `<main>`, `<footer>`), a
 * skip link that jumps focus to `<main>` for keyboard users (WCAG 2.4.1), and a
 * primary navigation whose active link is marked with `aria-current="page"`.
 * Full page content is filled in by tasks 13.2-13.5 through the nested
 * `<Outlet />`.
 */
export function Layout() {
  const navLinkClass = ({ isActive }: { isActive: boolean }) =>
    [
      'px-3 py-2 rounded-md text-sm font-medium',
      isActive
        ? 'bg-blue-700 text-white'
        : 'text-blue-800 hover:bg-blue-100 dark:text-blue-200 dark:hover:bg-blue-900',
    ].join(' ');

  return (
    <div className="min-h-screen flex flex-col">
      <a className="skip-link" href="#main-content">
        Skip to main content
      </a>

      <header className="border-b border-gray-200 dark:border-gray-800">
        <div className="mx-auto max-w-5xl px-4 py-4 flex items-center justify-between gap-4">
          <span className="text-lg font-semibold">ShelfSentry AI</span>
          <nav aria-label="Primary">
            <ul className="flex gap-1">
              <li>
                <NavLink to="/" end className={navLinkClass}>
                  Inspection
                </NavLink>
              </li>
              <li>
                <NavLink to="/results" className={navLinkClass}>
                  Results
                </NavLink>
              </li>
              <li>
                <NavLink to="/history" className={navLinkClass}>
                  History
                </NavLink>
              </li>
              <li>
                <NavLink to="/evaluation" className={navLinkClass}>
                  Evaluation
                </NavLink>
              </li>
            </ul>
          </nav>
        </div>
      </header>

      <main id="main-content" className="flex-1">
        <div className="mx-auto max-w-5xl px-4 py-8">
          <Outlet />
        </div>
      </main>

      <footer className="border-t border-gray-200 dark:border-gray-800">
        <div className="mx-auto max-w-5xl px-4 py-4 text-sm text-gray-500">
          ShelfSentry AI - active-perception shelf inspection.
        </div>
      </footer>
    </div>
  );
}
