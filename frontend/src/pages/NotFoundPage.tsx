import { Link } from 'react-router-dom';

/** 404 fallback route. Keeps the app shell navigable for unknown paths. */
export function NotFoundPage() {
  return (
    <section aria-labelledby="page-heading">
      <h1 id="page-heading" className="text-2xl font-semibold">
        Page not found
      </h1>
      <p className="mt-2 text-gray-600 dark:text-gray-300">
        The page you requested does not exist.
      </p>
      <p className="mt-4">
        <Link className="text-blue-700 underline" to="/">
          Go to Inspection
        </Link>
      </p>
    </section>
  );
}
