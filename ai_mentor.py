from openai import OpenAI
import os


def get_ai_advice(emotion):
    # Initialize the client correctly (OpenRouter base + your key)
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENAI_API_KEY")  # reads from environment
    )

    prompt = f"I’m feeling {emotion.lower()}. Please give a short, comforting message."

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a kind and empathetic AI mentor."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"⚠️ AI error: {str(e)}"
