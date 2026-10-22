# Sample inputs: provenance and license

## sample_shelf.png

- **Source:** Own work. Generated programmatically by
  `shelfsentry.samples.generate_sample_shelf_png()` using only NumPy and
  OpenCV 5. No external image was downloaded, copied, scraped, or derived.
- **Content:** A synthetic store-shelf scene — a plain gray background with
  three shelf rows of evenly spaced "product" boxes and two intentional empty
  gaps — drawn with fixed coordinates and colors (fully deterministic).
- **License:** Public domain / own work (released under
  [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/)). Because the
  ShelfSentry AI project authored this image from scratch, there is no
  third-party copyright or license to honor; it is trivially clean to bundle
  and redistribute with the project and its demo/submission artifacts.

The checked-in `sample_shelf.png` is produced by the generator above. It can be
regenerated deterministically at any time with
`shelfsentry.samples.ensure_samples()`, which yields byte-identical bytes.
