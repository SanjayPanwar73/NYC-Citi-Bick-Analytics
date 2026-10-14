# Deploying the dashboard

The pipeline (1 GB download, DuckDB, modeling) runs on **your computer**. What you deploy is the **dashboard + the small result files**
it reads (`reports/real/<range>/...`). The raw/processed data never goes to GitHub or a server.

## Option A - Streamlit Community Cloud (free, easiest)
1. Finish a real run locally (`python run_all.py status` shows every stage done) and open the dashboard once to check it.
2. Commit the result snapshot (the `.gitignore` already allows it): `git add reports/real .gitignore app DEPLOY.md` then
   `git status` - you should see `reports/real/<range>/data/*.csv|json|parquet|joblib` and `figures/*.png`; there must be **no** `data/` folder.
3. Pin scikit-learn in `app/requirements.txt` to the version in `reports/real/<range>/data/run_manifest.json` (`environment.packages`).
4. `git commit` and `git push`.
5. https://share.streamlit.io -> sign in with GitHub -> **Create app** -> repository, branch `main`, main file path `app/dashboard.py`.
   Under **Advanced settings** choose Python 3.12 (or the version you trained with). **Deploy**. You get `https://<name>.streamlit.app`.

## Other hosts
* **Hugging Face Spaces** (Streamlit SDK, free): create a Space, push the repo, set `app_file: app/dashboard.py` in the README header.
* **Render / Railway / Fly.io**: run `python -m streamlit run app/dashboard.py --server.port $PORT --server.address 0.0.0.0` from the repo root.
* GitHub Pages / Vercel cannot run Streamlit (they only serve static sites).

## Things to know
* Updating results: rerun locally -> commit the changed files in `reports/real/...` -> push; the app redeploys automatically.
* The Live tab calls the public GBFS feed from the server; it works without any key.
* Free tiers sleep after inactivity and have limited memory; the dashboard only loads small files, so this is fine.
* Add attribution to the README (data: Citi Bike / NYC bike share) and check the data-use terms on the Citi Bike system-data page.
* Do not commit secrets. None are needed. The run manifest holds package versions and counts only.
