-------------------- Git push------------------------------------------------
git init
git remote add origin https://github.com/namurugan/assignment_mlOps.git
git add .
git config --global user.name "namurugan"
git config --global user.email "2023ac05878@wilp.bits-pilani.ac.in"
git commit -m "Initial commit without mlruns, ml artifacts and venv1"
git commit - m "code commit"
git push -u origin master

---------------------------DVC-----------------------------------------
dvc init
dvc add data/raw/california_housing_raw.csv
dvc data/processed/california_housing_processed.csv
git add data/raw/california_housing_raw.csv.dvc data/processed/california_housing_processed.csv.dvc .gitignore
git commit -m "Add raw and processed datasets tracked by DVC"
git push

----------------------deploy in docker -----------------------
docker build -t california-housing-api .
docker run -d -p 8000:8000 -v ${PWD}/logs:/app/logs 3a9de75c82f3a8174934af0a759877b3135696b22de60de8c12c50fee735835a

-------to test api without docker ----------------
uvicorn app.main:app --reload

pip install prometheus_client

