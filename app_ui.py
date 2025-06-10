import streamlit as st
import requests
import json
import time

st.title("LuminAI")
st.write("Hey Prajwal! How can I help with your project today?")

# Initialize session state variables if they don't exist
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'session_id' not in st.session_state:
    st.session_state.session_id = None
if 'available_sessions' not in st.session_state:
    st.session_state.available_sessions = {}

# Hidden user ID, set default to "Prajwal"
user_id = "Prajwal"

# Settings in sidebar
with st.sidebar:
    st.header("Settings")
    project = st.text_input("Project", "ProjectA")
    
    # Session management
    st.header("Sessions")
    
    # Fetch available sessions for the user and project
    try:
        response = requests.get(f"http://localhost:5001/api/sessions?user_id={user_id}&project={project}")
        if response.status_code == 200:
            st.session_state.available_sessions = response.json().get("sessions", {})
        else:
            st.warning("Could not fetch sessions")
    except Exception as e:
        st.warning("Error connecting to server")
    
    # Create a new session button
    if st.button("Create New Session"):
        try:
            response = requests.post("http://localhost:5001/api/sessions/create", 
                                    data={"user_id": user_id, "project": project})
            if response.status_code == 201:
                session_data = response.json()
                st.session_state.session_id = session_data["session_id"]
                st.session_state.chat_history = []
                st.success("New session created!")
                st.rerun()  # Refresh to update session list
            else:
                st.error("Failed to create new session")
        except Exception as e:
            st.error("Connection error. Check if the server is running.")
    
    # Display available sessions
    if st.session_state.available_sessions:
        st.subheader("Your Sessions")
        sessions = st.session_state.available_sessions
        session_names = {sid: session.get("name", f"Session {i+1}") 
                        for i, (sid, session) in enumerate(sessions.items())}
        
        # If we have a current session ID but it doesn't match any loaded session,
        # reset it to None (this can happen when switching projects)
        if st.session_state.session_id and st.session_state.session_id not in session_names:
            st.session_state.session_id = None
            st.session_state.chat_history = []
        
        # Select a session
        selected_session_name = st.selectbox(
            "Select a session", 
            list(session_names.values()),
            index=list(session_names.values()).index(
                session_names.get(st.session_state.session_id, list(session_names.values())[0])
            ) if st.session_state.session_id in session_names else 0
        )
        
        # Find the session ID from the name
        selected_session_id = next(
            (sid for sid, name in session_names.items() if name == selected_session_name), 
            None
        )
        
        # Update session state if selection changed
        if selected_session_id and selected_session_id != st.session_state.session_id:
            st.session_state.session_id = selected_session_id
            st.session_state.chat_history = []  # Clear local history when switching
            st.rerun()  # Refresh to ensure consistency
    
    # Session actions
    if st.session_state.session_id:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear History"):
                try:
                    response = requests.post("http://localhost:5001/api/sessions/clear", 
                                           data={"user_id": user_id, 
                                                "project": project, 
                                                "session_id": st.session_state.session_id})
                    if response.status_code == 200:
                        st.session_state.chat_history = []
                        st.success("History cleared!")
                        st.rerun()
                    else:
                        st.error("Failed to clear history")
                except Exception as e:
                    st.error("Connection error")
        
        with col2:
            if st.button("Delete Session"):
                try:
                    response = requests.post("http://localhost:5001/api/sessions/delete", 
                                           data={"user_id": user_id, 
                                                "project": project, 
                                                "session_id": st.session_state.session_id})
                    if response.status_code == 200:
                        st.session_state.session_id = None
                        st.session_state.chat_history = []
                        st.success("Session deleted!")
                        st.rerun()                   
                    else:
                        st.error("Failed to delete session")
                except Exception as e:
                    st.error("Connection error")

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
        data = {
            "user_id": user_id, 
            "project": project, 
            "query_text": query, 
            "stream": "true"
        }
        
        # Add session_id if it exists
        if st.session_state.session_id:
            data["session_id"] = st.session_state.session_id
        
        with requests.post("http://127.0.0.1:5001/api/query", data=data, stream=True) as response:
            if response.status_code == 200:
                current_session_id = st.session_state.session_id
                
                for line in response.iter_lines():
                    if line:
                        try:
                            line_text = line.decode('utf-8')
                            if line_text.startswith("data: "):
                                json_str = line_text[6:]  # Remove "data: " prefix
                                chunk_data = json.loads(json_str)
                                
                                # Check if this is a session_id update
                                if 'session_id' in chunk_data:
                                    current_session_id = chunk_data['session_id']
                                    # Update session state with the new ID if none exists
                                    if not st.session_state.session_id:
                                        st.session_state.session_id = current_session_id
                                elif 'chunk' in chunk_data:
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
                
                # Store the session ID if it was updated
                if current_session_id != st.session_state.session_id:
                    st.session_state.session_id = current_session_id
                    st.rerun()  # Refresh to update UI with new session
            else:
                error_msg = "I couldn't process that request. Want to try again?"
                message_placeholder.markdown(error_msg)
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
    except Exception as e:
        error_msg = "Oops! There was a connection problem. Make sure your backend server is running."
        message_placeholder.markdown(error_msg)
        st.session_state.chat_history.append({"role": "assistant", "content": error_msg})