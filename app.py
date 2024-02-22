"""Streamlit app to generate Tweets."""

# Import from standard library
import logging
import json

# Import from 3rd party libraries
import streamlit as st
import streamlit.components.v1 as components
import streamlit_analytics

# Import modules
import ai

# Configure logger
logging.basicConfig(format="\n%(asctime)s\n%(message)s", level=logging.INFO, force=True)

# Define functions
def run_identify(object: str = ""):
    """Generate Tweet text."""
    if st.session_state.n_requests >= 5:
        st.session_state.text_error = "Too many requests. Please wait a few seconds before generating another Analysis."
        logging.info(f"Session request limit reached: {st.session_state.n_requests}")
        st.session_state.n_requests = 1
        return
    
    st.session_state.components = ""
    st.session_state.text_error = ""

    if not object:
        st.session_state.text_error = "Please enter an object to classify."
        return

    with text_spinner_placeholder:
        with st.spinner("Please wait for the analysis to be generated..."):
            classifier = ai.Classifier()
            st.session_state.n_requests += 1
            streamlit_analytics.start_tracking()

            st.session_state.components = (
                classifier.classify(object=object)
                )

            logging.info(
                f"Inputs: {object}\n"
                f"Output: {st.session_state.components}"
            )


# Configure Streamlit page and state
st.set_page_config(page_title="Object Classifier", page_icon="🤖")


if "components" not in st.session_state:
    st.session_state.components = ""
if "text_error" not in st.session_state:
    st.session_state.text_error = ""
if "n_requests" not in st.session_state:
    st.session_state.n_requests = 0


# Force responsive layout for columns also on mobile
st.write(
    """<style>
    [data-testid="column"] {
        width: calc(50% - 1rem);
        flex: 1 1 calc(50% - 1rem);
        min-width: calc(50% - 1rem);
    }
    </style>""",
    unsafe_allow_html=True,
)


# Render Streamlit page
streamlit_analytics.start_tracking()
st.title("Object Classifier App")
st.markdown(f"""
                The Object Classifier App is a powerful tool for classifying textual descriptions of objects using cutting-edge artificial intelligence models. With this application, users can input an object, and the AI model will analyze and providing detailed information about the components or attributes of the object.
            """)

object = st.text_input(
    label="What is the object?",
    placeholder="Canned Tuna"
)

st.button(
    label="Identify Components!",
    type="primary",
    on_click=run_identify,
    args=[object],
)
text_spinner_placeholder = st.empty()

if st.session_state.text_error:
    st.error(st.session_state.text_error)

if st.session_state.components:
    st.markdown("""---""")
    st.text_area(label="Classification", value=st.session_state.components, height=82)