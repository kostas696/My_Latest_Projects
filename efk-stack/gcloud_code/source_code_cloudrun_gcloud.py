import functions_framework
import base64
import vertexai
from vertexai.generative_models import GenerativeModel

@functions_framework.http
def pubsub_handler(request):
    try:
        envelope = request.get_json()
        if not envelope or "message" not in envelope:
            raise ValueError("Invalid Pub/Sub message")

        pubsub_message = envelope["message"]
        data = base64.b64decode(pubsub_message["data"]).decode("utf-8").strip()

        print(f"[LOG] Received error: {data}")

        # Use a supported region
        vertexai.init(project="log-anomaly-detector", location="us-central1")

        # Use available Gemini model in that region
        model = GenerativeModel("gemini-2.0-flash-lite-001")

        prompt = f"""
        This is a system or application log error: "{data}".
        Explain the error clearly, identify its root cause, and suggest a resolution.
        Be concise and specific, no more than 5 sentences.
        """

        response = model.generate_content(prompt)
        print(f"[AI Response]: {response.text}")

        return "OK", 200

    except Exception as e:
        print(f"[ERROR] {e}")
        return f"Error: {e}", 500
