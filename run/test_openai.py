from dotenv import load_dotenv
from knowornot import KnowOrNot

load_dotenv()


kon = KnowOrNot()

kon.add_openai(default_model="gpt-4o", default_embedding_model="text-embedding-3-large")
output = kon.get_client().prompt("Hello there!")
