import streamlit as st
import requests

st.title("Project Knowledge Assistant")

# User inputs
col1, col2 = st.columns(2)
with col1:
    user_id = st.text_input("User ID", "user1")
    project = st.text_input("Project", "ProjectA")
with col2:
    meeting_type = st.selectbox("Meeting Type", ["Technical Meeting", "KT Session", "Townhall Meeting"])

# File upload
uploaded_file = st.file_uploader("Upload transcript", type=['txt'])
if uploaded_file is not None and st.button("Process Transcript"):
    files = {"transcript": uploaded_file}
    data = {"user_id": user_id, "project": project, "meeting_type": meeting_type}
    response = requests.post("http://localhost:5000/api/upload", files=files, data=data)
    if response.status_code == 200:
        st.success("Upload successful!")
    else:
        st.error("Upload failed")

# Chat interface
st.divider()
st.subheader("Ask about your project")
query = st.text_input("Your question:")
if query and st.button("Submit"):
    with st.spinner("Generating response..."):
        data = {"user_id": user_id, "project": project, "query_text": query}
        response = requests.post("http://localhost:5000/api/query", data=data)
        if response.status_code == 200:
            st.write(response.json()["response"])
        else:
            st.error("Query failed")