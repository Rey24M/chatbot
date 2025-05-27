import streamlit as st
from huggingface_hub import InferenceClient
from datetime import datetime
#import json

# client stuff
MODEL = "HuggingFaceH4/zephyr-7b-beta"
#API line
HF_TOKEN = "API token here"  
client = InferenceClient(model=MODEL, token=HF_TOKEN)

if "input_counter" not in st.session_state:
    st.session_state.input_counter = 0


if "history" not in st.session_state:
    st.session_state.history = []
    st.session_state.context = {
        "summary": "",
        "topics": [],
        "last_seen": datetime.now().strftime("%Y-%m-%d %H:%M")
    }


def update_context(user_input, bot_response):
    """Maintains conversation context without summarization"""
    new_ctx = f"User: {user_input}\nBot: {bot_response}\n"
    st.session_state.context["summary"] = (new_ctx + st.session_state.context["summary"])[:2000]  
    
    

# UI
st.title("🧠Chatbot")


user_input = st.text_input("Your message:", key="user_input", 
                          placeholder="Type here and press Enter...")



if user_input:
    try:
        # Store user message
        st.session_state.history.append({"sender": "You", "text": user_input, "time": datetime.now().strftime("%H:%M")})
        
        # prompt
        prompt = f"""
        [Context]
        Previous topics: {', '.join(st.session_state.context['topics'][-3:]) or 'None'}
        Last conversation: {st.session_state.context['summary'][:300]}...
        
        [Current Message]
        User: {user_input}
        
        Assistant:"""
        
        with st.spinner("Thinking..."):
            response = client.text_generation(
        prompt=prompt,
        max_new_tokens=256,
        temperature=0.7,
        repetition_penalty=1.2
    ).strip()
            if "Assistant:" in response:
                response = response.split("Assistant:")[-1].strip()
    
            response = response.split('\n')[0].strip()
            
            # Store bot response
            st.session_state.history.append({"sender": "Bot", "text": response, "time": datetime.now().strftime("%H:%M")})
            
            
            update_context(user_input, response)
            
            
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")

# Display messages
st.markdown("### Conversation")
for msg in reversed(st.session_state.history):  # Newest first
    css_class = "user-bubble" if msg["sender"] == "You" else "bot-bubble"
    st.markdown(f"""
    <div class="{css_class}">
        <strong>{msg['sender']}:</strong> {msg['text']}
        <div style='font-size: 0.8em; color: #666;'>{msg['time']}</div>
    </div>
    """, unsafe_allow_html=True)

# sidebar (extra stuff)
with st.sidebar:
    st.subheader("Conversation Memory")
    st.write(f"📅 Last active: {st.session_state.context['last_seen']}")
    
    
    if st.button("🧹 Clear Memory"):
        st.session_state.context = {"summary": "", "topics": [], "last_seen": datetime.now().strftime("%Y-%m-%d %H:%M")}
        st.experimental_rerun()
    

# CSS for Bubbles
st.markdown("""
<style>
    .user-bubble {
        background-color: #0078D4;
        color: white;
        padding: 10px 15px;
        border-radius: 18px 18px 0 18px;
        margin: 8px 0;
        margin-left: 20%;
        max-width: 75%;
    }
    .bot-bubble {
        background-color: #f0f0f0;
        color: black;
        padding: 10px 15px;
        border-radius: 18px 18px 18px 0;
        margin: 8px 0;
        margin-right: 20%;
        max-width: 75%;
    }
</style>
""", unsafe_allow_html=True)
