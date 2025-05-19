import streamlit as st
import requests
import json
import time

st.title("LuminAI")
st.write(f"Hey Prajwal! How can I help with your project today?")

# Initialize session state for chat history if it doesn't exist
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Hidden user ID, set default to "Prajwal"
user_id = "Prajwal"

# Settings in sidebar with meeting type default "Unknown"
with st.sidebar:
    st.header("Settings")
    project = st.text_input("Project", "ProjectA")
    meeting_type = st.selectbox("Meeting Type", 
                               ["Unknown", "Technical Meeting", "KT Session", "Townhall Meeting"],
                               index=0)  # Set "Unknown" as default
    
    # File upload
    st.header("Upload the video")
    uploaded_file = st.file_uploader("Select file", type=['txt'])
    if uploaded_file is not None and st.button("Process Video"):
        files = {"transcript": uploaded_file}
        data = {"user_id": user_id, "project": project, "meeting_type": meeting_type}
        try:
            response = requests.post("http://localhost:5000/api/upload", files=files, data=data)
            if response.status_code == 200:
                st.success("Got it! Your video has been processed.")
            else:
                st.error(f"Hmm, something went wrong with the upload.")
        except Exception as e:
            st.error("Connection error. Check if the server is running.")

# Main chat area
st.header("Chat")

# Display chat history
for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.chat_message("user").write(message["content"])
    else:
        st.chat_message("assistant").write(message["content"])

# Chat input
query = st.chat_input("What would you like to know?")
if query:
    # Display user message
    st.chat_message("user").write(query)
    
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": query})
    
    # Create a placeholder for the streaming response
    assistant_message = st.chat_message("assistant")
    full_response = ""
    message_placeholder = assistant_message.empty()
    
    # Stream the response
    try:
        data = {"user_id": user_id, "project": project, "query_text": query, "stream": "true"}
        with requests.post("http://localhost:5000/api/query", data=data, stream=True) as response:
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        try:
                            line_text = line.decode('utf-8')
                            if line_text.startswith("data: "):
                                json_str = line_text[6:]  # Remove "data: " prefix
                                chunk_data = json.loads(json_str)
                                chunk = chunk_data.get("chunk", "")
                                full_response += chunk
                                # Update the message with the accumulated response
                                message_placeholder.markdown(full_response + "▌")
                                time.sleep(0.01)  # Small delay for smoother appearance
                        except Exception as e:
                            print(f"Error processing chunk: {e}")
                
                # Once done, update with the final response without the cursor
                message_placeholder.markdown(full_response)
                # Add the complete response to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": full_response})
            else:
                error_msg = "I couldn't process that request. Want to try again?"
                message_placeholder.markdown(error_msg)
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
    except Exception as e:
        error_msg = "Oops! There was a connection problem. Make sure your backend server is running."
        message_placeholder.markdown(error_msg)
        st.session_state.chat_history.append({"role": "assistant", "content": error_msg})

# Add option to clear chat history with casual language
if st.session_state.chat_history and st.sidebar.button("Start Fresh"):
    st.session_state.chat_history = []
    st.rerun()