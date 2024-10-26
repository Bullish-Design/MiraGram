import os
from dotenv import load_dotenv


load_dotenv()

db_name = os.getenv("POSTGRES_DB")
db_user = os.getenv("POSTGRES_USER")
db_pass = os.getenv("POSTGRES_PASSWORD")


db_url = "postgresql://" + db_user + ":" + db_pass + "@localhost/" + db_name


logdir = os.getenv("LOGDIR")
code_output_dir = os.getenv("CODE_OUTPUT_DIR")
local_model_name = os.getenv("LOCAL_MODEL_NAME")
local_model_url = os.getenv("LOCAL_MODEL_URL")
