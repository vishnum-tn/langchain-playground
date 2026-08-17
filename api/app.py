import requests
import streamlit as st


def get_response(input_text, endpoint):
    """Send request to LangServe endpoint and return the response."""
    try:
        response = requests.post(
            f"http://localhost:8000/{endpoint}/invoke",
            json={'input': {'topic': input_text}},
            timeout=120,
        )

        if response.status_code != 200:
            st.error(f"Server returned {response.status_code}: {response.text[:300]}")
            return None

        data = response.json()
        output = data.get("output", "")

        if isinstance(output, dict):
            return output.get("content", str(output))

        return str(output)

    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to server. Make sure api/client.py is running on port 8000.")
        return None
    except requests.exceptions.JSONDecodeError:
        st.error(f"Server returned non-JSON response: {response.text[:200]}")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return None


st.title('LangChain Demo with Gemma & Gemini')
input_text = st.text_input("Write an essay on (Gemma / llama.cpp)")
input_text1 = st.text_input("Write a poem on (Gemini)")

if input_text:
    with st.spinner("Generating essay with Gemma..."):
        result = get_response(input_text, "essay")
        if result:
            st.write(result)

if input_text1:
    with st.spinner("Generating poem with Gemini..."):
        result = get_response(input_text1, "poem")
        if result:
            st.write(result)
