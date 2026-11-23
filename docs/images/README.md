# Screenshots for the README

Save these four screenshots here, with exactly these file names (PNG, full browser width, dashboard running):

| File name | Page | What to show |
|---|---|---|
| `overview.png` | Executive overview | Takeaways box + the two rows of KPI cards |
| `customer_profile.png` | Customer 360 profile | Risk gauge, profile fields and the "Why this risk score?" chart |
| `explainable_ai.png` | Explainable AI | Importance chart + "How a factor moves predictions" |
| `retention.png` | Retention strategy | Priority tiers + the call list |

On Windows: open the page, press **Win + Shift + S**, select the area, then paste into Paint and save as PNG here.

Then, in the main `README.md`, delete the two marker lines (`<!-- SCREENSHOTS: ...` and `END -->`) around the screenshot table, and push:

```bash
git add docs/images README.md
git commit -m "Add dashboard screenshots"
git push
```
