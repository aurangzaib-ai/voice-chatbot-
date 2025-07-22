from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.output_parser import StrOutputParser
from streamlit_mic_recorder import speech_to_text
from gtts import gTTS
from gtts.lang import tts_langs
import streamlit as st
import os

# Streamlit app ka page configuration set karta hai
st.set_page_config(page_title="AI Voice/Text Assistant", page_icon="🤖")

# App ka title aur subtitle show karta hai
st.title("AI Voice/Text Assistant 🎙️💬")
st.subheader("Interact in Urdu with Real-Time Voice or Text Input")
st.image("https://www.purespeechtechnology.com/wp-content/uploads/2020/04/voice-assistant-enterprise-conversational-ai.jpg", use_column_width=True)

# Google API key jo AI model ke liye use hoti hai
api_key = "AIzaSyCgmyTU9CKXnzgjD0R7Zt-IOBKY759gyLA"

# Prompt template jo AI ko input deta hai aur chat history ko track karta hai
prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            "You are a helpful AI assistant. Please always respond to user queries in Pure Urdu language."
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
)

# Streamlit chat message history ko store karne ke liye
msgs = StreamlitChatMessageHistory(key="langchain_messages")

# Google Generative AI model ko load karta hai
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)

# Chain setup jo AI se response le kar output ko parse karta hai
chain = prompt | model | StrOutputParser()

# Chain ko history k sath run karta hai taake AI pichlay messages ko bhi samjh sakay
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: msgs,
    input_messages_key="question",
    history_messages_key="chat_history",
)

# Available languages ko check karta hai jo TTS support karta hai
langs = tts_langs().keys()

# Option deta hai ke user voice ya text input de sakay
option = st.radio("Choose input method:", ('Voice', 'Text'))

if option == 'Voice':
    st.write("Press the button and start speaking in Urdu:")
    with st.spinner("Converting Speech To Text..."):
        # Speech ko Urdu main text main convert karta hai
        text = speech_to_text(
            language="ur", use_container_width=True, just_once=True, key="STT"
        )
elif option == 'Text':
    # Text input ka option show karta hai
    text = st.text_input("Enter your question in Urdu:")

# Jab user ne text provide kiya hai (ya to voice se convert ho ya direct input)
if text:
    # Human message ko chat main display karta hai
    st.chat_message("human").write(text)
    
    # AI ka response chat main dikhata hai
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        config = {"configurable": {"session_id": "any"}}
        response = chain_with_history.stream({"question": text}, config)

        for res in response:
            full_response += res or ""
            message_placeholder.markdown(full_response + "|")
            message_placeholder.markdown(full_response)

    # Text ko speech mein convert kar ke wapas play karna
    with st.spinner("Converting Text To Speech..."):
        tts = gTTS(text=full_response, lang="ur")
        tts.save("output.mp3")
        st.audio("output.mp3")

else:
    st.warning("Please provide your input via voice or text.")

