# Deployment guide: GitHub + Streamlit Community Cloud

The goal is a public link (e.g. `https://customer360-ai.streamlit.app`) to put on your CV and LinkedIn.

## How deployment works for this project
- **Code** lives on GitHub. **Data, the database, models and reports are not committed** (see `.gitignore`).
- On the **first start**, the app sees no data. It downloads the IBM Telco dataset and runs the whole pipeline once, which takes about a minute (`app/data.py → ensure_pipeline()`). Visitors who arrive during that minute wait for the same build instead of starting another.
- Community Cloud's disk is temporary. If the app is rebooted or wakes from sleep, the first-start build simply runs again.
- **Public demo mode** (`C360_DEMO_MODE = "1"`) turns off saving uploads to the database, so no visitor's data is ever stored. Scoring uploads still works, in memory only.

## Step 1: Put the code on GitHub (once)
1. Create a free account at <https://github.com> if you don't have one.
2. Click **New repository**. Name it `customer360-ai`, make it **Public**, and **don't** add a README (the project already has one).
3. In PowerShell, inside the project folder:
```powershell
git init
git add .
git status                      # check: no venv/, data/raw/, database/*.db, .env listed
git commit -m "Customer360 AI: analytics, segmentation, churn prediction, dashboard"
git branch -M main
git remote add origin https://github.com/<your-username>/customer360-ai.git
git push -u origin main
```
If Git asks who you are, run `git config --global user.name "Your Name"` and `git config --global user.email "you@example.com"` first.

4. On GitHub, open the **Actions** tab. The **tests** workflow runs the full test suite on Python 3.12 and 3.14. A green tick means it passed.

## Step 2: Deploy on Streamlit Community Cloud
1. Go to <https://share.streamlit.io> and sign in with GitHub.
2. Click **Create app → Deploy a public app from GitHub**.
3. Fill in:
   - **Repository:** `<your-username>/customer360-ai`
   - **Branch:** `main`
   - **Main file path:** `app/streamlit_app.py`
   - **App URL:** choose a name, e.g. `customer360-ai`
4. Open **Advanced settings**:
   - **Python version:** choose the same version as your laptop (3.14) if it's offered, otherwise 3.13 or 3.12. The code supports 3.10+.
   - **Secrets:** paste
     ```toml
     C360_DEMO_MODE = "1"
     ```
5. Click **Deploy**. The first build installs the libraries (a few minutes). The first visit then shows *"First start: downloading the dataset and building the models…"* for about a minute.

## Updating the app
Commit and push to `main`. Community Cloud picks up the change and restarts the app automatically.
```powershell
git add .
git commit -m "Describe the change"
git push
```

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Build fails installing packages | A library has no build for the cloud's Python version | Redeploy and choose another Python version in Advanced settings (it can only be changed by deleting and redeploying) |
| "Automatic setup failed" on first visit | The dataset download was blocked or timed out | Reboot the app from the Community Cloud menu (⋮ → Reboot) |
| Theme looks like default Streamlit | `.streamlit/config.toml` not committed | Check it's in the GitHub repo at the top level of the project |
| Changes don't appear | The app hasn't restarted | ⋮ → Reboot app |

## Running locally (unchanged)
```powershell
python scripts/run_all.py --download
streamlit run app/streamlit_app.py
```
Set `C360_AUTO_BOOTSTRAP=0` to turn off the automatic first-start build, or `C360_DEMO_MODE=1` to try demo mode locally.
