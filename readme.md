# Setup environment

- Install python

- Test python works

```cmd
python --version
```

- Create virtual environment (venv)

```cmd
python -m venv sbg
```

- Activate virtual environment


```cmd
cd <venv dir>
sbg\Scripts\activate
```

- Install packages from requirements-sbg.txt


```cmd
python -m pip install -r requirements-sbg.txt 
```

# Usage

- Add herbarium image (.jpg) to img/ folder

- Activate python environment
```cmd
cd <venv dir>
sbg\Scripts\activate
```

- Launch marimo notebook

```
marimo edit exercise-2.py
```

- Run the notebook, specifying the image's filename under 'image'. 
