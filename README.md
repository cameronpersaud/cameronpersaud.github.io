# cameronpersaud.nyc

Personal site of Cameron Persaud. Static HTML, no build step.

## Files
- `index.html`: the whole site (styles and scripts are inline)
- `404.html`: shown by GitHub Pages for missing pages
- `assets/`: headshot, logo, favicon, artwork, and the CV PDF
- `CNAME`: custom domain for GitHub Pages
- `.nojekyll`: tells GitHub Pages to serve files as they are

## Publish on GitHub Pages
1. Put these files at the root of the repository (replace the old site's files).
2. In the repository: Settings → Pages → Source: "Deploy from a branch", Branch: `main`, folder `/ (root)`.
3. Custom domain: `cameronpersaud.nyc` (already set by `CNAME`). Turn on "Enforce HTTPS".

## Updating the CV
Replace `assets/Cameron_Persaud_CV.pdf` with the new PDF, keeping the same file name.
