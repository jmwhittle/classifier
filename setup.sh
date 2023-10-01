# pyenv install 3.9.9
# pyenv virtualenv 3.9.9 classifier

# pyenv activate classifier
poetry install
poetry run python -m ipykernel install --user --name classifier