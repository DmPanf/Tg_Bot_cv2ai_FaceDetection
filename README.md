## Install Redis

```bash
sudo apt install -y redis
```

```bash
sudo systemctl enable redis-server --now
```

## Install poetry

```bash
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
```

## Clone repo

```bash
git clone {PROJECT_URL} $HOME/Documents/
```

```bash
cd $HOME/Documents/{PROJECT_NAME}
```

## Install project

```bash
poetry install --no-dev
```

```bash
poetry env use system
```

## Edit env

```bash
mv .env.dist .env ; \
nano .env
```