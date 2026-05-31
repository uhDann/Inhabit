# Hosting the interactive viewer on GitHub Pages

The viewers are static three.js pages that load committed `.ply` meshes, so they
work on GitHub Pages with **no build step**.

## One important gotcha

**Do NOT put the viewer meshes in Git LFS.** GitHub Pages serves the LFS
*pointer file*, not the actual mesh, so the viewer would silently fail to load.
The decimated meshes under `runs/` are therefore committed **directly** (they're
small — ~7–13 MB each, force-added past the `*.ply` gitignore). That's the
correct setup for Pages.

## Enable it (repo Settings → Pages)

1. Push this repo to GitHub (public).
2. **Settings → Pages → Build and deployment → Source: "Deploy from a branch".**
3. Branch: **`main`**, folder: **`/ (root)`**. Save.
4. After ~1 minute the site is live at:
   - Landing gallery: `https://<user>.github.io/<repo>/`
   - 4-method viewer: `https://<user>.github.io/<repo>/viewer/mesh_compare.html`
   - Recon vs GT: `https://<user>.github.io/<repo>/viewer/replica_room0.html`

The relative `../runs/*.ply` paths in the viewers resolve correctly when Pages
serves from the repo root.

## Or via the CLI (if you have `gh` authenticated)

```bash
gh repo create <repo> --public --source=. --push
gh api -X POST repos/<user>/<repo>/pages -f source[branch]=main -f source[path]=/ 
```

The landing page is `index.html` at the repo root.
