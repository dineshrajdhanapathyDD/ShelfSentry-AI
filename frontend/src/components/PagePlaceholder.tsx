interface PagePlaceholderProps {
  title: string;
  description: string;
  task: string;
}

/**
 * A minimal, accessible placeholder for a page whose full content is delivered
 * by a later task (13.2-13.5). Renders a labeled heading and a short note so
 * the route is navigable and screen-reader announced before the real page
 * exists. Replaced wholesale when its task lands.
 */
export function PagePlaceholder({ title, description, task }: PagePlaceholderProps) {
  return (
    <section aria-labelledby="page-heading">
      <h1 id="page-heading" className="text-2xl font-semibold">
        {title}
      </h1>
      <p className="mt-2 text-gray-600 dark:text-gray-300">{description}</p>
      <p className="mt-4 text-sm text-gray-500">
        Placeholder - full page delivered in {task}.
      </p>
    </section>
  );
}
