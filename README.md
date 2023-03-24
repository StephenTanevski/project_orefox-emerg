How to contribute

If you want to work on a feature which is completly different than existing branches, create a new branch and work on it. For example, if you want to work on data science, then create a new branch named data_science, then create a folder to that branch named data_science and then put relevant files and folders inside it. Make sure all branch, file and folder names do not contain spaces. Instead you can use underscores (recommended) or hyphens.



How to run this app in Local Machine (Linux Recommended)

Clone this project
Create and activate virtual environment
Goto 'web' directory and install requirements with through pip pip install -r requirements.txt
Create a .env file in 'web' directory and put all the default variables from .env.structure.txt
Clone this fork of the Spirit package outside of this repo: https://github.com/GeorgeKandamkolathy/Spirit
Within this repo take the spirit file and replace the installed spirit folder within the virtual environment lib/python3.8/site-packages.
Make sure you have the latest project version and run python manage.py spiritinstall
Migrate the database with python manage.py migrate
Create superuser with python manage.py createsuperuser then follow the instruction
Now start the application with python manage.py runserver and visit this link http://127.0.0.1:8000/
Go to admin panel with http://127.0.0.1:8000/admin or which one you've setup
Then setup all the "Processes" with valid codes otherwise app won't work. For now there are only two codes required "OSNACA", "ALS"
Then you are ready to use the application...Go to site http://127.0.0.1:8000/ and use this application.
Note: Processes is a wrong naming convention. We will use appropriate name in later version.
