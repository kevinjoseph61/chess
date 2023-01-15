# Chess-Gang

ASGI Django Server to play Chess in real-time using Django Channels and Redis.

## `master branch`
Master branch is for developement Django server to be run on local system (Windows). Follow these steps to use:  
1. Install all the dependencies by running `pipenv install` (install pipenv first)  
2. Ensure you have Redis running on `port #6379` for it to work correctly. This can be achieved using Docker (on Windows) by running the command `docker run -p 6379:6379 -d redis:5`  
3. Finally run server using `python manage.py runserver`  

## `prod branch`
Prod branch was made keeping in mind the end result of deployment on Heroku using the free-tier. You can view the version that I have deployed [here](https://chess-gang.herokuapp.com/).  
`IMPORTANT! Please Read: Free tier is no longer available on Heroku so this link will not work as the dyno is down. The production steps will still work nevertheless`  
Follow these steps to deploy:  
1. Clone this branch and create a new application on Heroku  
2. Add the Heroku Redis addon to the application (free-tier available)  
3. Link the cloned repo to your application and deploy
