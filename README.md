## Install Redis

```bash
sudo apt install -y redis
```

```bash
sudo systemctl enable redis-server --now
```

## Install supervisor

```bash
sudo apt install -y supervisor
```

```bash
sudo systemctl enable supervisor --now
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

## Edit env

```bash
mv .env.dist .env ; \
nano .env
```

## Run project

Supervisor will keep the project running