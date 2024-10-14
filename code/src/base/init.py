import os
from dotenv import load_dotenv


load_dotenv()

db_name = os.getenv("POSTGRES_DB")
db_user = os.getenv("POSTGRES_USER")
db_pass = os.getenv("POSTGRES_PASSWORD")


db_url = "postgresql://" + db_user + ":" + db_pass + "@localhost/" + db_name
