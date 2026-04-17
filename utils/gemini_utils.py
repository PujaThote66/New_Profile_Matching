import google.generativeai as genai


# ------------------------------------------------------
# ✅ Select compatible Gemini model (NO GLOBAL STATE)
# ------------------------------------------------------
def get_compatible_model(api_key: str):
    """
    Dynamically selects the first Gemini model that supports text generation.
    The API key is provided per request (from Streamlit UI).
    """

    genai.configure(api_key=api_key)

    for m in genai.list_models():
        if "generateContent" in m.supported_generation_methods:
            return genai.GenerativeModel(m.name)

    raise RuntimeError("No compatible Gemini model found for generateContent.")


# ------------------------------------------------------
# ✅ Interview Question Generator (UPDATED)
# ------------------------------------------------------
def generate_interview_questions(
    jd: str,
    resume: str,
    api_key: str
) -> str:

    if not jd.strip():
        raise ValueError("Job Description is empty.")

    if not resume.strip():
        raise ValueError("Candidate resume is empty.")

    prompt = f"""
You are a senior technical interviewer.

Job Description:
{jd}

Candidate Resume:
{resume}

Generate:
1. 5 Behavioral interview questions
2. 5 L1 Technical interview questions

Rules:
- Do NOT include answers
- Avoid advanced system design questions
- Keep questions clear and practical
- Use EXACTLY this format:

Behavioral Questions:
1.
2.
3.
4.
5.

L1 Technical Questions:
1.
2.
3.
4.
5.
"""

    try:
        model = get_compatible_model(api_key)
        response = model.generate_content(prompt)

    except Exception as e:
        raise RuntimeError(f"Gemini API call failed: {str(e)}")

    if not response or not hasattr(response, "text"):
        raise RuntimeError("Gemini returned an invalid response.")

    if not response.text.strip():
        raise RuntimeError("Gemini returned an empty response.")

    return response.text.strip()
