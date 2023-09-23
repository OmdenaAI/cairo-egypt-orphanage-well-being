# Getting Started with This Django Web App

## Prerequisites 🛠️

Before you dive in, ensure you have the following tools installed and ready:

1. **Python:** You'll need Python 3.6 or a newer version. Download it from the [Python website](https://www.python.org/downloads/).

2. **PostgreSQL:** Our app will elegantly dance with PostgreSQL, our database partner. Fetch it from [PostgreSQL's lair](https://www.postgresql.org/download/).

## Setting Up Your Dev Playground 💻

1. **Clone this repository**

2. **Virtual Environment:** Keep your project's dependencies tidy by creating a virtual environment. In your terminal, navigate to your project folder:

    ```sh
    cd /cairo-egypt-orphanage-well-being/src/tasks/task-9-develop-web-app/monitorApp
    ```

3. **Create a Virtual Haven:** Execute the following command to craft a virtual environment named "venv":

    ```sh
    python3 -m venv venv
    ```

4. **Activate Your Realm:** Enchant your terminal with the virtual environment's powers:

    - **On Windows:**

        ```sh
        venv\Scripts\activate
        ```

    - **On macOS and Linux:**

        ```sh
        source venv/bin/activate
        ```

5. **Backup the Database:** Before proceeding, if you have received an `orphansdb.sql` file, you can restore the database from it. Assuming you have PostgreSQL installed, you can use the following command to restore the database:

    ```sh
    psql -U postgres -d orphansdb -f orphansdb.sql
    ```

    Replace `-U postgres` with your PostgreSQL username if it's different, and make sure to adjust the database name (`orphansdb`) accordingly.

## Breathing Life into Your Project 🚀

1. **Install Dependencies:** While inside your virtual haven, install the required packages using pip:

    ```sh
    pip install -r requirements.txt
    ```

2. **Database Sorcery:** Open `monitorApp/settings.py` and navigate to the `DATABASES` section. Mold the settings to align with PostgreSQL:

    ```python
    DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.postgresql',
            'NAME': 'orphansdb', #create this database in postgres to avoid errors
            'USER': 'postgres', #make sure to use the account with which postgres was configured
            'PASSWORD': '12345', #check if your password is right
            'HOST': 'localhost',
            'PORT': '5432',
        }
    }
    ```

    Replace `'orphansdb'`, `'postgres'`, and `'12345'` with your PostgreSQL database's credentials.

3. **Make Migrations:** Create initial database migrations using the following command:

    ```sh
    python manage.py makemigrations
    ```

4. **Summon Migrations:** Cast the migration spell in your terminal to set up the database:

    ```sh
    python manage.py migrate
    ```

5. **Set Up Email:** For the sake of functionalities like forgot password this server can send emails to set that up. Open `monitorApp/settings.py` and navigate to

    ```python
    '''Email configuration starts here'''
    EMAIL_BACKEND ='django.core.mail.backends.smtp.EmailBackend'
    EMAIL_HOST ='smtp.gmail.com'
    EMAIL_PORT = 587
    EMAIL_USE_TLS = True
    EMAIL_HOST_USER = os.environ.get('EMAIL_USER') #create an environment variable for your email or put your email here
    EMAIL_HOST_PASSWORD = os.environ.get('EMAIL_PASS') #create an environment variable for your app password set up in gmail or put it here
    '''Email configuration ends here'''
    ```

## Igniting Your Development Server 🔥

1. **Creating a Superuser:** Command your terminal to fashion an admin superuser:

    ```sh
    python manage.py createsuperuser
    ```

2. **Launching the Mothership:** Propel your app into the virtual cosmos:

    ```sh
    python manage.py runserver
    ```

    Behold! This app now lives at [http://127.0.0.1:8000/](http://127.0.0.1:8000/), while the admin realm resides at [http://127.0.0.1:8000/admin/](http://127.0.0.1:8000/admin/).

## Accessing App Pages 🌐

You can access various pages of the app using the following URLs:

1. **Dashboard:** [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
    - Path: `/`
    - Functionality: The main dashboard of the application.

2. **Cameras:** [http://127.0.0.1:8000/cameras/](http://127.0.0.1:8000/cameras/)
    - Path: `/cameras/`
    - Functionality: View cameras.

3. **New Camera:** [http://127.0.0.1:8000/new_camera/](http://127.0.0.1:8000/new_camera/)
    - Path: `/new_camera/`
    - Functionality: Add a new camera.

4. **ML Script:** [http://127.0.0.1:8000/mlscript/](http://127.0.0.1:8000/mlscript/)
    - Path: `/mlscript/`
    - Functionality: Access machine learning script.

5. **Upload:** [http://127.0.0.1:8000/upload/](http://127.0.0.1:8000/upload/)
    - Path: `/upload/`
    - Functionality: Upload videos for inference.

These URLs allow you to navigate through the different functionalities of the application. Happy exploring!


Happy coding and may your code be bug-free! 🚀
