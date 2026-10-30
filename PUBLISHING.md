# Putting BrainStack 3D online with GitHub Pages

The `docs/` folder holds a complete browser version of BrainStack 3D. It runs entirely in the
visitor's web browser, so GitHub Pages can host it for free. Nobody's images are uploaded
anywhere: the page reads the files the person picks and does all the work on their own computer.

Your site will be at **`https://<your-github-username>.github.io/Brainstack/`**.

## 1. Put the new files in your repository

1. Unzip `BrainStack3D.zip`.
2. Copy everything inside the unzipped `BrainStack3D` folder into your existing repository
   folder (`Documents/GitHub/Brainstack`), replacing files when asked.
   - Do **not** delete the hidden `.git` folder in `Brainstack`; that is what links the
     folder to GitHub.
   - You should now have `Brainstack/docs/index.html`.

## 2. Commit and push

**With GitHub Desktop**
1. Open GitHub Desktop. It lists the new and changed files.
2. In the box at the bottom left, type a summary such as `Add browser version`.
3. Click **Commit to main**.
4. Click **Push origin** at the top.

**Or in the VS Code terminal** (inside the `Brainstack` folder)
```
git add -A
git commit -m "Add browser version"
git push
```

## 3. Turn on GitHub Pages (once)

1. On github.com, open your `Brainstack` repository.
2. Go to **Settings → Pages** (in the left sidebar).
3. Under **Build and deployment → Source**, choose **Deploy from a branch**.
4. Set **Branch** to `main` and the folder to **`/docs`**, then click **Save**.
5. Wait one to two minutes, then refresh the page. A box at the top shows
   **"Your site is live at …"** with the link.

## 4. Updating the site later

Any change you commit and push to `docs/` on `main` is live about a minute later. You can
follow the progress in the repository's **Actions** tab.

## Trying it on your own computer first

The page uses JavaScript modules, which browsers block when a file is opened by
double-clicking it. Start a tiny local web server instead:

```
cd docs
python3 -m http.server 8000
```

Then open <http://localhost:8000> in your browser. Press Ctrl+C in the terminal to stop it.

## Good to know

- **Public repository = public code.** Anyone can see the code (not your data). Free GitHub
  accounts need a public repository for Pages.
- **Browsers:** current Chrome, Edge, Firefox and Safari all work. Chrome or Edge is fastest.
- **Big stacks:** everything happens in browser memory. Use *Balanced (600 px)* unless you need
  more detail. A stack of 20–40 sections works comfortably on a normal laptop.
- **What the browser version does:** everything in the five steps — loading JPG/PNG/TIFF
  (8- and 16-bit, multi-channel), cutting plane and order, scale, dye channel,
  alignment (including mirrored slices), the 3D dye-in-brain view, and downloads.
  - Downloads: dye report (CSV), picture (PNG), dye shape and brain surface (STL).
  - The desktop app additionally exports the aligned 3D stack as a TIFF for Fiji, and has
    extra views: point cloud and full volume rendering.
- **Testing:** `node tests/web_e2e.js <image-folder> <output-folder> [plane] [spacing µm] [tissue µm]`
  drives the whole browser app automatically. It needs Node.js and
  `npm install playwright`.
