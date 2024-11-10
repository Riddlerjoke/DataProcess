import streamlit as st


def main():
    # Use the full page instead of a narrow central column
    st.set_page_config(layout="wide")

    st.title("ML-Ops.bzh")
    st.subheader(
        "Machine learning operations (MLOps) is the discipline of AI model delivery. It focuses on the ML model's development, deployment, monitoring, and lifecycle management. MLOps enables data scientists and IT operations teams to collaborate and increase the pace of model development and deployment via monitoring, validation, and governance of machine learning models.")


if __name__ == "__main__":
    main()
