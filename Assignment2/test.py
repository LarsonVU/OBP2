import streamlit as st

def main():
    st.title("Streamlit App with Two Tabs")
    
    tab1, tab2 = st.tabs(["Tab 1", "Tab 2"])
    
    with tab1:
        st.header("Welcome to Tab 1")
        st.write("This is the content for the first tab.")
        st.button("Click me on Tab 1")
    
    with tab2:
        st.header("Welcome to Tab 2")
        st.write("This is the content for the second tab.")
        st.text_input("Enter some text on Tab 2")

if __name__ == "__main__":
    main()