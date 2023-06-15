# Chess-Gang

ASGI Django Server to play Chess in real-time using Django Channels and Redis.

## `master branch`

Master branch is for developement Django server to be run on local system (Windows). Follow these steps to use:

1. Install all the dependencies by running `pipenv install` (install pipenv first)
2. Ensure you have Redis running on `port #6379` for it to work correctly. This can be achieved using Docker (on Windows) by running the command `docker run -p 6379:6379 -d redis:5`
3. Finally run server using `python manage.py runserver`

## `render branch`

Render branch was created so that render.com can be a drop-in replacement for deploying freely in leiu of Heroku. View my deployment [here](https://chess-gang.onrender.com/) (Takes a few seconds to startup).  
Follow these steps to deploy:

1. Create a Redis application on render and note down the internal URL. Should look like `redis://red-[UNIQUE SEQUENCE]:6379`
2. Clone this repo and create a new web application on Render with your `render` branch
3. Create the following environment variables for application: `PYTHON_VERSION` : `3.8.17`, `REDIS_URL`: [URL you noted down], `SECRET_KEY `: [Use the generate option to create a secure one]
4. Build command should be `sh build.sh` and start command should be `pipenv run daphne pychess.asgi:application --bind 0.0.0.0 -v2`

## `prod branch` (Not free anymore [Heroku])

Prod branch was made keeping in mind the end result of deployment on Heroku using the free-tier. You can view the version that I have deployed [here](https://chess-gang.herokuapp.com/).  
`IMPORTANT! Please Read: Free tier is no longer available on Heroku so this link will not work as the dyno is down. The production steps will still work nevertheless`  
Follow these steps to deploy:

1. Clone this branch and create a new application on Heroku
2. Add the Heroku Redis addon to the application (free-tier available)
3. Link the cloned repo to your application and deploy
