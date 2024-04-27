import os
import logging
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI, HTTPException
from prompt import Prompt
from promptHistory import PromptHistory
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.DEBUG)


with open("hue.txt") as file:
    hue_prompt = file.read()

with open("gen.txt") as file:
    gen_prompt = file.read()

load_dotenv(find_dotenv())

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

model = "gpt-3.5-turbo"

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/gen/")
async def gen(prompt_history: PromptHistory):
    if not prompt_history:
        raise HTTPException(status_code=400, detail="Input list cannot be empty")

    try:
        history_list = [{"role": "system", "content": gen_prompt}]
        history_list += [{"role": item.role, "content": item.content} for item in prompt_history.prompt_history]
        logging.debug(history_list)
        completion = client.chat.completions.create(
            model=model,
            temperature=0.3,
            messages=history_list

        )
        return {"prompt": completion.choices[0].message.content}
    except Exception as e:
        # Handle potential errors during API call
        print(f"Error during completion: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/hue/")
async def hue(prompt: Prompt):
    print(prompt)
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt cannot be empty")

    try:
        completion = client.chat.completions.create(
            model=model,
            temperature=0.7,
            messages=[{"role": "system", "content": hue_prompt}, {"role": prompt.role, "content": prompt.content}]
        )
        return {"prompt": completion.choices[0].message.content}
    except Exception as e:
        # Handle potential errors during API call
        print(f"Error during completion: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
