import os

from dotenv import load_dotenv
from pyrogram import Client, filters

from script.intent_classification import get_intention, labels

load_dotenv()

api_id = os.getenv('API_ID')
api_hash = os.getenv('API_HASH')
bot_token = os.getenv('BOT_TOKEN')

app = Client("my_account", api_hash=api_hash,
             api_id=api_id, bot_token=bot_token)

@app.on_message(filters.private)
async def hello(client, message):

    intention_pred, _ = get_intention(message.text)
    await message.reply(f"You want to {labels[intention_pred[0]]}")



app.run()
