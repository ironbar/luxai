# Challenge workflow

## Start of the challenge

1. Create a repository for the code using cookiecutter
2. Add dates to the calendar
2. Download rules of the challenge
3. Bookmark challenge folder on file explorer
4. Create a Google keep label for tasks and ideas of the challenge
5. Download the challenge data
6. Create a conda environment for the challenge and add it to jupyter

```bash
conda create -n luxai pytest rope pylint tqdm numpy pandas scikit-learn ipython ipykernel coverage
source activate luxai
python -m ipykernel install --user --name $CONDA_DEFAULT_ENV --display-name "Python ($CONDA_DEFAULT_ENV)"
conda env export > environment.yml
```

8. Work on the challenge
8. Use TDD methodology whenever possible, this will save time because errors
won't be propagated along the challenge.
9. Have an aprentice attitude, collaborate on the forum, I have a lot to learn
from Kaggle.

## End of the challenge

1. Prepare a report with a summary of the aproach to the challenge
2. Download the Google keep tasks to the repository in pdf format
3. Delete the tasks on google keep and the label
4. Delete unnecessary data
5. Update the environment yml
