import os
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI, HTTPException
from promptHistory import PromptHistory
load_dotenv(find_dotenv())

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

model = "gpt-3.5-turbo"
temp = 0.3

app = FastAPI()


@app.post("/gen/")
async def process_list(prompt_history: PromptHistory):

    if not prompt_history:
        raise HTTPException(status_code=400, detail="Input list cannot be empty")

    try:
        completion = client.chat.completions.create(
            model=model,
            temperature=temp,
            messages=[{"role": item.role, "content": item.content} for item in prompt_history.prompt_history]
        )
        return {"prompt": completion.choices[0].message.content}
    except Exception as e:
        # Handle potential errors during API call
        print(f"Error during completion: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")




