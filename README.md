# Shelter Adoption Prediction: Austin Animal Center

Predict, **at the moment a dog or cat arrives at the shelter**, how likely it is to be adopted.

ML Zoomcamp midterm project.

## Problem description

The Austin Animal Center is the City of Austin's municipal animal shelter, taking in well over ten thousand animals
a year. Every animal eventually leaves the shelter in one of a few ways: it is **adopted**, returned to its owner, transferred to a partner
rescue, or (rarely) euthanized or dies in care.

Shelter staff have limited time for adoption outreach: featured listings, social media posts, adoption events,
foster placements. Today they decide where to spend that effort by experience. A model that scores each new arrival
with its probability of adoption lets them:

- **Flag at-risk animals on day one**: low-probability animals can be prioritized for outreach, fostering, or
  early contact with partner rescues, instead of waiting until they've been in the shelter for weeks.
- **Plan capacity**: summed probabilities across today's intake give a rough estimate of upcoming adoptions.

**Task:** binary classification. For each shelter stay, predict `adopted = 1` if the outcome is `Adoption`, and `0` for
any other outcome (transfer, return to owner, Rto-Adopt, euthanasia, died, ...).

The model only uses information available **at intake**: animal type, intake type and condition, sex and
spay/neuter status, age, breed, color, month, and how many times the animal has been to the shelter before.

> **What "adopted" means here:** the shelter itself adopted the animal out. Some breeds that are popular as pets score
> low (e.g. Shih Tzu, Pug) because they usually leave through other routes: they are reclaimed by owners or
> transferred to breed-specific rescues. The model predicts the shelter's own adoption outcome, not general demand.

## Data

Two public datasets from the City of Austin Open Data Portal, committed in `data/`:

| File | Rows | Contents |
|---|---|---|
| `data/aac_intakes.csv` | ~174k | One row per intake: date, intake type and condition, animal type, sex, age, breed, color |
| `data/aac_outcomes.csv` | ~174k | One row per outcome: date, outcome type, ... |

Sources: [Austin Animal Center Intakes](https://data.austintexas.gov/Health-and-Community-Services/Austin-Animal-Center-Intakes/wter-evkm)
and [Austin Animal Center Outcomes](https://data.austintexas.gov/Health-and-Community-Services/Austin-Animal-Center-Outcomes/9t4d-g238).
The committed copies cover October 2013 to May 2025. No download step is needed.

### Preparing the data

- **Matching intakes to outcomes.** The two files share an `Animal ID` but no stay ID, and some animals come in
  many times (one dog has 33 stays). Each intake is matched to the animal's first outcome after it, unless that
  outcome happens after the animal's *next* intake (then it belongs to the later stay).
- **Scope.** Dogs and cats only. Dropped intakes that aren't adoption candidates (`Wildlife`, `Public Assist`,
  `Euthanasia Request`) and outcomes with unknown results (`Missing`, `Stolen`, `Lost`).
  This leaves **152,517 stays, 53% adopted**.
- **No leakage.** Only intake-time columns are features. Outcome-time columns (sex and age at outcome) and the
  animal's **name** are excluded: the name column holds the animal's final name, and shelter-assigned names
  strongly signal adoption.

### Train / validation / test split

The model would be used on future intakes, so the split is **by time**, not random:

| Split | Intake dates | Stays | Adoption rate |
|---|---|---|---|
| Train | Oct 2013 - Jul 2019 | 91,511 | 48% |
| Validation | Jul 2019 - Apr 2022 | 30,503 | 56% |
| Test | Apr 2022 - May 2025 | 30,503 | 64% |

The adoption rate rises over time, so models are compared on **ROC AUC**, which measures how well the model ranks
animals and isn't affected by that shift.

## Approach

All of this is in [`austin_animal_center_EDA.ipynb`](austin_animal_center_EDA.ipynb).

### Feature engineering

| Feature | Built from | Example |
|---|---|---|
| `breed_type` | `Breed` | `Beagle` -> pure, `Beagle Mix` -> mix, `Beagle/Boxer` -> cross |
| `primary_breed` | `Breed` | first breed listed, truncated names fixed (`Chesa Bay Retr` -> `Chesapeake Bay Retriever`); breeds with < 200 training stays -> `Other Dog` / `Other Cat` |
| `age_days` | `Age upon Intake` | `2 years` -> 730 |
| `sex`, `is_fixed` | `Sex upon Intake` | `Neutered Male` -> Male, 1 |
| `color_primary`, `is_multicolor` | `Color` | `Brown Tabby/White` -> Brown Tabby, 1 |
| `intake_month` | intake date | seasonality (kitten season) |
| `prior_stays` | stay history | number of earlier stays for this animal |
| `animal_type`, `intake_type`, `intake_condition` | as-is | rare conditions -> `Other` |

The raw `Breed` column has 2,656 distinct values. It isn't free text: it follows a fixed format (`X`, `X Mix`,
`X/Y`), so parsing it reduces it to 252 primary breeds, then 46 common breeds plus `Other`.

### EDA highlights

- Pure-breed dogs are adopted **less** often (39%) than mixes (50%) and two-breed crosses (56%). Breed type barely
  matters for cats.
- Dog adoption rates by breed range from 67% (Black Mouth Cur) to 14% (Shih Tzu).
- Age is the strongest single signal: younger animals are adopted far more often.

### Models

Four models, each with a small hyperparameter grid, compared on validation AUC:

| Model | Best parameters | Validation AUC |
|---|---|---|
| **Gradient boosting** (`HistGradientBoostingClassifier`) | learning_rate=0.05, max_leaf_nodes=15 | **0.743** |
| Random forest | max_depth=20, min_samples_leaf=5 | 0.739 |
| Decision tree | max_depth=10, min_samples_leaf=200 | 0.729 |
| Logistic regression | C=0.001 | 0.695 |

The final gradient boosting model, retrained on train + validation, scores **0.774 AUC on the test years
(2022-2025)**.

**Feature importance** (permutation importance on the final model): age matters most by a wide margin, followed by
intake type, intake condition, and sex. Breed is a moderate signal; color, month and multicolor coats contribute little.

## Repository structure

```
├── README.md
├── austin_animal_center_EDA.ipynb   # data prep, EDA, feature importance, model selection
├── train.py                         # trains the final model -> model.bin
├── predict.py                       # Flask web service serving model.bin
├── predict-test.py                  # sends a sample animal to the service
├── model.bin                        # trained model (output of train.py)
├── Pipfile, Pipfile.lock            # dependencies (pipenv)
├── Dockerfile, .dockerignore        # container for the web service
└── data/                            # raw intake and outcome CSVs
```

## How to run

### 1. Set up the environment

Dependencies are managed with [Pipenv](https://pipenv.pypa.io/), which creates a virtual environment from
`Pipfile.lock`. Requires Python 3.12.

```bash
pip install pipenv
pipenv install --dev      # --dev adds Jupyter and requests, for the notebook and the test script
pipenv shell              # activate the environment
```

### 2. Run the notebook (optional)

```bash
jupyter notebook austin_animal_center_EDA.ipynb
```

Run all cells; the whole notebook takes a few minutes.

### 3. Train the model

```bash
python train.py
```

Takes about a minute. Prints validation and test AUC and writes `model.bin`. A trained `model.bin` is already
committed, so this step is only needed to reproduce it.

### 4. Run the web service locally

```bash
gunicorn --bind=0.0.0.0:9696 predict:app
```

(On Windows, gunicorn doesn't work; use `python predict.py` instead.)

In a second terminal (also inside `pipenv shell`):

```bash
python predict-test.py
```

### 5. Run the web service with Docker

```bash
docker build -t adoption-prediction .
docker run -it --rm -p 9696:9696 adoption-prediction
```

Then, in a second terminal: `python predict-test.py`

## Using the API

`POST /predict` with one animal's intake record as JSON:

```bash
curl -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{
    "animal_type": "Dog",
    "intake_type": "Stray",
    "intake_condition": "Normal",
    "sex_upon_intake": "Intact Male",
    "age_upon_intake": "2 years",
    "breed": "Labrador Retriever Mix",
    "color": "Black/White",
    "intake_datetime": "2025-03-01T10:00:00",
    "prior_stays": 0
  }'
```

Response:

```json
{"adopted": true, "adoption_probability": 0.56}
```

| Field | Values |
|---|---|
| `animal_type` | `Dog`, `Cat` |
| `intake_type` | `Stray`, `Owner Surrender`, `Abandoned` |
| `intake_condition` | e.g. `Normal`, `Injured`, `Sick`, `Nursing`, `Neonatal`, `Aged` |
| `sex_upon_intake` | `Intact Male`, `Intact Female`, `Neutered Male`, `Spayed Female`, `Unknown` |
| `age_upon_intake` | `<number> <day/week/month/year>(s)`, e.g. `3 weeks`, `1 year` |
| `breed` | shelter format, e.g. `Pit Bull Mix`, `Beagle/Boxer`, `Domestic Shorthair` |
| `color` | shelter format, e.g. `Brown Tabby/White`, `Black` |
| `intake_datetime` | ISO date-time of intake |
| `prior_stays` | number of earlier stays at the shelter (0 for a first visit) |

Unrecognized breeds, colors and conditions are treated as `Other`, so any value in the shelter's format works.

## Cloud deployment

_Coming soon._

## Limitations

- The adoption rate drifts upward over time (48% -> 64%). AUC is stable, but the probabilities themselves run
  low on recent data, so the 0.5 cutoff for `adopted` is conservative.
- Animals that were still in the shelter when the data was exported have no outcome yet and are excluded, which
  may slightly over-represent quick outcomes in the most recent months.
