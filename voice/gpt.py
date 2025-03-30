import os
import json
import openai
openai.api_key = ""

def gpt_request(prompt):

    # Initial text
    p = prompt

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f'{p}'}
        ]
    )

    print(response)
    
    r = response['choices'][0]['message']['content']

    print(r)

    return r
