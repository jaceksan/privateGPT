import os
import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = "org-FGVSBhEOLC3mgOJhR0kXIO1n"


def load_chain():
    """Logic for loading the chain you want to use should go here."""
    # llm = OpenAI(temperature=0, model_name="gpt-3.5-turbo-0613")
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0613")

    chain = ConversationChain(llm=llm)
    return chain


def chat_gpt():
    if "generated" not in st.session_state:
        st.session_state["generated"] = []

    if "past" not in st.session_state:
        st.session_state["past"] = []

    try:
        chain = load_chain()

        user_input = st.text_area("Ask GoodData a question:")

        if st.button("Submit Query", type="primary"):
            output = chain.run(input=user_input)
            st.session_state.past.append(user_input)
            st.session_state.generated.append(output)

        if st.session_state["generated"]:
            for i in range(len(st.session_state["generated"]) - 1, -1, -1):
                message(st.session_state["generated"][i], key=str(i))
                message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
    except openai.error.AuthenticationError as e:
        st.write("OpenAI unknown authentication error")
        st.write(e.json_body)
        st.write(e.headers)
