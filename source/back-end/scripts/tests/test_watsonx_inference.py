import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from app.core.config import settings
from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai import Credentials

def test_inference():
    try:
        credentials = Credentials(
            url=settings.WATSONX_URL,
            api_key=settings.WATSONX_APIKEY,
            project_id=settings.WATSONX_PROJECT_ID
        )

        model = Model(
            model_id="ibm/granite-13b-chat-v2",
            credentials=credentials,
            params={
                "decoding_method": "greedy",
                "max_new_tokens": 100
            }
        )

        prompt = "What is IBM Watsonx.ai and what is it used for?"
        result = model.generate(prompt)
        generated_text = result["results"][0]["generated_text"]

        print("Watsonx inference successful.")
        print("Generated answer:")
        print(generated_text)

    except Exception as e:
        print(f"Error during inference: {e}")

if __name__ == "__main__":
    test_inference()
