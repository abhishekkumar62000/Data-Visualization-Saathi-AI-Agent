import matplotlib.pyplot as plt
import os
import json
import re
import sys
import io
import contextlib
import warnings
from typing import Optional, List, Any, Tuple
from PIL import Image
import streamlit as st
import pandas as pd
import base64
from io import BytesIO
from together import Together
from e2b_code_interpreter import Sandbox
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

pattern = re.compile(r"```python\n(.*?)\n```", re.DOTALL)

def code_interpret(e2b_code_interpreter: Sandbox, code: str) -> Optional[List[Any]]:
    with st.spinner('Executing code in E2B sandbox...'):
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                exec = e2b_code_interpreter.run_code(code)

        if stderr_capture.getvalue():
            print("[Code Interpreter Warnings/Errors]", file=sys.stderr)
            print(stderr_capture.getvalue(), file=sys.stderr)

        if stdout_capture.getvalue():
            print("[Code Interpreter Output]", file=sys.stdout)
            print(stdout_capture.getvalue(), file=sys.stdout)

        if exec.error:
            print(f"[Code Interpreter ERROR] {exec.error}", file=sys.stderr)
            return None
        return exec.results

def match_code_blocks(llm_response: str) -> str:
    match = pattern.search(llm_response)
    if match:
        code = match.group(1)
        return code
    return ""

def chat_with_llm(e2b_code_interpreter: Sandbox, user_message: str, dataset_path: str) -> Tuple[Optional[List[Any]], str]:
    # Conversational AI Assistant with Contextual Memory
    system_prompt = f"""You're a Python data scientist and data visualization expert. You are given a dataset at path '{dataset_path}'.\nYou need to analyze the dataset and answer the user's query with a response and you run Python code to solve them.\nIMPORTANT: Always use the dataset path variable '{dataset_path}' in your code when reading the CSV file."""
    messages = [
        {"role": "system", "content": system_prompt}
    ]
    # Add previous chat history as alternating user/assistant messages
    for q, a in st.session_state.chat_history:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
    # Add current user message
    messages.append({"role": "user", "content": user_message})

    with st.spinner('Getting response from Together AI LLM model...'):
        client = Together(api_key=st.session_state.together_api_key)
        response = client.chat.completions.create(
            model=st.session_state.model_name,
            messages=messages,
        )

        response_message = response.choices[0].message
        python_code = match_code_blocks(response_message.content)
        
        if python_code:
            code_interpreter_results = code_interpret(e2b_code_interpreter, python_code)
            return code_interpreter_results, response_message.content
        else:
            st.warning(f"Failed to match any Python code in model's response")
            return None, response_message.content

def upload_dataset(code_interpreter: Sandbox, uploaded_file) -> str:
    dataset_path = f"./{uploaded_file.name}"
    
    try:
        code_interpreter.files.write(dataset_path, uploaded_file)
        return dataset_path
    except Exception as error:
        st.error(f"Error during file upload: {error}")
        raise error


def main():
    """Main Streamlit application."""

    # --- Custom Dark Colorful UI/UX Styling ---
    st.markdown(
        """
        <style>
        /* App background and main container */
        .stApp {
            background: linear-gradient(135deg, #232526 0%, #414345 100%);
            color: #f8f8f2;
            font-family: 'Segoe UI', 'Roboto', 'Arial', sans-serif;
        }
        /* Title with animation */
        .main-title {
            font-size: 2.8rem;
            font-weight: bold;
            background: linear-gradient(90deg, #ff512f, #dd2476, #1fa2ff, #12c2e9, #f64f59);
            background-size: 200% auto;
            color: #fff;
            background-clip: text;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: gradient-move 3s linear infinite;
            text-align: center;
            margin-bottom: 0.5em;
        }
        @keyframes gradient-move {
            0% {background-position: 0% 50%;}
            100% {background-position: 100% 50%;}
        }
        /* Subtitle */
        .subtitle {
            color: #ffd700;
            font-size: 1.2rem;
            text-align: center;
            margin-bottom: 1.5em;
        }
        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: linear-gradient(135deg, #141e30 0%, #243b55 100%);
            color: #fff;
            border-radius: 18px 0 0 18px;
            box-shadow: 2px 0 16px #0004;
        }
        /* Tabs */
        div[data-baseweb="tab-list"] {
            background: linear-gradient(90deg, #232526 0%, #1fa2ff 100%);
            border-radius: 12px;
            padding: 0.2em 0.5em;
        }
        button[data-baseweb="tab"] {
            color: #fff;
            font-weight: 600;
            background: none;
            border-radius: 8px 8px 0 0;
            transition: background 0.3s, color 0.3s;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            background: linear-gradient(90deg, #ff512f 0%, #dd2476 100%);
            color: #fff;
            box-shadow: 0 2px 8px #0002;
        }
        /* Buttons */
        .stButton>button {
            background: linear-gradient(90deg, #ff512f 0%, #dd2476 100%);
            color: #fff;
            border: none;
            border-radius: 8px;
            font-weight: 600;
            box-shadow: 0 2px 8px #0002;
            transition: background 0.2s, transform 0.2s;
        }
        .stButton>button:hover {
            background: linear-gradient(90deg, #1fa2ff 0%, #12c2e9 100%);
            color: #ffd700;
            transform: scale(1.04);
        }
        /* Dataframe and code blocks */
        .stDataFrame, .stTable, .stCodeBlock, .stMarkdown pre {
            background: #232526 !important;
            color: #ffd700 !important;
            border-radius: 10px;
        }
        /* Chat bubbles (override for both user and AI) */
        div[style*='background:linear-gradient(90deg,#232526,#414345)'],
        div[style*='background:linear-gradient(90deg,#141e30,#243b55)'] {
            animation: fadeIn 0.7s;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        /* Download buttons */
        .stDownloadButton>button {
            background: linear-gradient(90deg, #12c2e9 0%, #c471f5 100%);
            color: #fff;
            border-radius: 8px;
            font-weight: 600;
            transition: background 0.2s, color 0.2s;
        }
        .stDownloadButton>button:hover {
            background: linear-gradient(90deg, #f64f59 0%, #c471f5 100%);
            color: #ffd700;
        }
        /* Inputs */
        .stTextInput>div>div>input, .stTextArea textarea {
            background: #232526;
            color: #fff;
            border: 1.5px solid #ffd700;
            border-radius: 8px;
        }
        /* Checkbox */
        .stCheckbox>label>div:first-child {
            border: 2px solid #ffd700;
        }
        /* Subheaders */
        .stSubheader {
            color: #12c2e9;
        }
        /* Markdown links */
        .stMarkdown a {
            color: #ffd700;
            text-decoration: underline;
        }
        /* Animations for tab content */
        .block-container {
            animation: fadeIn 0.8s;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="main-title">📊 AI Data Visualization Agent</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Upload your dataset and ask questions about it!</div>', unsafe_allow_html=True)

    # Load .env variables
    load_dotenv()

    # Initialize session state variables
    if 'together_api_key' not in st.session_state:
        st.session_state.together_api_key = os.environ.get('TOGETHER_API_KEY', '')
    if 'e2b_api_key' not in st.session_state:
        st.session_state.e2b_api_key = os.environ.get('E2B_API_KEY', '')
    if 'model_name' not in st.session_state:
        st.session_state.model_name = ''
    # Chat history: list of (question, answer) tuples
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    with st.sidebar:
        # Only show model selection dropdown
        model_options = {
            "Meta-Llama 3.1 405B": "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
            "DeepSeek V3": "deepseek-ai/DeepSeek-V3",
            "Qwen 2.5 7B": "Qwen/Qwen2.5-7B-Instruct-Turbo",
            "Meta-Llama 3.3 70B": "meta-llama/Llama-3.3-70B-Instruct-Turbo"
        }
        st.session_state.model_name = st.selectbox(
            "Select Model",
            options=list(model_options.keys()),
            index=0  # Default to first option
        )
        st.session_state.model_name = model_options[st.session_state.model_name]

    # --- Main App Tabs ---
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if 'uploaded_file_name' not in st.session_state:
        st.session_state.uploaded_file_name = None
    if 'df' not in st.session_state:
        st.session_state.df = None
    if uploaded_file is not None and uploaded_file.name != st.session_state.uploaded_file_name:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.session_state.uploaded_file_name = uploaded_file.name
        except pd.errors.EmptyDataError:
            st.error("The uploaded CSV file is empty or invalid. Please upload a valid CSV file with data.")
            st.session_state.df = None
            st.session_state.uploaded_file_name = None
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            st.session_state.df = None
            st.session_state.uploaded_file_name = None
    df = st.session_state.df

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Data Analysis",
        "🤖 AI Chatbot",
        "🧠 Auto Insights & Anomaly Detection",
        "📚 Data Storytelling & Dashboards"
    ])

    with tab1:
        if df is not None:
            st.write("Dataset:")
            show_full = st.checkbox("Show full dataset")
            if show_full:
                st.dataframe(df)
            else:
                st.write("Preview (first 5 rows):")
                st.dataframe(df.head())

            # --- Insight Suggestions ---
            st.subheader("Insight Suggestions")
            suggestions = []
            # Suggest columns with high missing values
            missing = df.isnull().sum()
            high_missing = missing[missing > 0].sort_values(ascending=False)
            if not high_missing.empty:
                suggestions.append(f"Columns with missing values: {', '.join(high_missing.index)}")
            # Suggest columns with few unique values (potential categories)
            few_uniques = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtype == 'object']
            if few_uniques:
                suggestions.append(f"Categorical columns: {', '.join(few_uniques)}")
            # Suggest numeric columns for correlation
            if len(df.select_dtypes(include='number').columns) > 1:
                suggestions.append("Try exploring correlations between numeric columns.")
            # Suggest outlier check
            for col in df.select_dtypes(include='number').columns:
                if df[col].max() > df[col].mean() + 3*df[col].std():
                    suggestions.append(f"Column '{col}' may have outliers.")
            # Default suggestion
            if not suggestions:
                suggestions.append("Try asking about trends, group comparisons, or summary statistics!")
            for s in suggestions:
                st.markdown(f"- {s}")

            # --- Automatic Data Profiling ---
            st.subheader("Automatic Data Profiling Report")
            st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
            st.write("**Column Types:**")
            st.write(pd.DataFrame({'Column': df.columns, 'Type': df.dtypes.values}))
            st.write("**Missing Values:**")
            st.write(df.isnull().sum())
            st.write("**Descriptive Statistics:**")
            st.write(df.describe(include='all'))
            numeric_cols = df.select_dtypes(include='number').columns
            cat_cols = df.select_dtypes(include='object').columns
            # Interactive visualizations with Plotly and Altair
            if len(numeric_cols) > 0:
                st.write("**Numeric Columns Distribution (Interactive):**")
                for col in numeric_cols:
                    fig = None
                    # Plotly histogram
                    fig = None
                    try:
                        import plotly.express as px
                        fig = px.histogram(df, x=col, nbins=20, title=f"Histogram of {col}")
                        st.plotly_chart(fig, use_container_width=True)
                        # Download option for Plotly
                        plotly_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
                        st.download_button(
                            label=f"Download {col} Histogram (HTML)",
                            data=plotly_html,
                            file_name=f"histogram_{col}.html",
                            mime="text/html"
                        )
                    except Exception as e:
                        st.warning(f"Plotly error for {col}: {e}")
                    # Altair histogram
                    try:
                        import altair as alt
                        alt_chart = alt.Chart(df).mark_bar().encode(
                            x=alt.X(col, bin=alt.Bin(maxbins=20)),
                            y='count()'
                        ).properties(title=f"Altair Histogram of {col}")
                        st.altair_chart(alt_chart, use_container_width=True)
                        st.download_button(
                            label=f"Download {col} Altair Chart (JSON)",
                            data=alt_chart.to_json(),
                            file_name=f"altair_histogram_{col}.json",
                            mime="application/json"
                        )
                    except Exception as e:
                        st.warning(f"Altair error for {col}: {e}")
            if len(cat_cols) > 0:
                st.write("**Top Categories (Interactive):**")
                for col in cat_cols:
                    top_vals = df[col].value_counts().head(10).reset_index()
                    top_vals.columns = [col, 'Count']
                    # Plotly bar
                    try:
                        import plotly.express as px
                        fig = px.bar(top_vals, x=col, y='Count', title=f"Top 10 {col}")
                        st.plotly_chart(fig, use_container_width=True)
                        plotly_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
                        st.download_button(
                            label=f"Download {col} Bar Chart (HTML)",
                            data=plotly_html,
                            file_name=f"bar_{col}.html",
                            mime="text/html"
                        )
                    except Exception as e:
                        st.warning(f"Plotly error for {col}: {e}")
                    # Altair bar
                    try:
                        import altair as alt
                        alt_chart = alt.Chart(top_vals).mark_bar().encode(
                            x=col,
                            y='Count'
                        ).properties(title=f"Altair Top 10 {col}")
                        st.altair_chart(alt_chart, use_container_width=True)
                        st.download_button(
                            label=f"Download {col} Altair Bar (JSON)",
                            data=alt_chart.to_json(),
                            file_name=f"altair_bar_{col}.json",
                            mime="application/json"
                        )
                    except Exception as e:
                        st.warning(f"Altair error for {col}: {e}")

            # --- Conversational Chat History ---
            st.subheader("Chat History")
            for i, (q, a) in enumerate(st.session_state.chat_history):
                st.markdown(f"**User:** {q}")
                st.markdown(f"**AI:** {a}")
                st.markdown("---")

            # Query input
            query = st.text_area("What would you like to know about your data?",
                                "Can you compare the average cost for two people between different categories?")

            if st.button("Analyze"):
                if not st.session_state.together_api_key or not st.session_state.e2b_api_key:
                    st.error("Please enter both API keys in the sidebar.")
                else:
                    with Sandbox(api_key=st.session_state.e2b_api_key) as code_interpreter:
                        # Upload the dataset
                        dataset_path = upload_dataset(code_interpreter, uploaded_file)

                        # Pass dataset_path to chat_with_llm
                        code_results, llm_response = chat_with_llm(code_interpreter, query, dataset_path)

                        # Add to chat history
                        st.session_state.chat_history.append((query, llm_response))

                        # Display LLM's text response
                        st.write("AI Response:")
                        st.write(llm_response)

                        # --- Enhanced Code Playground: Interactive and Outstanding ---
                        code_block = match_code_blocks(llm_response)
                        if code_block:
                            st.markdown("---")
                            st.markdown("### 🛠️ Code Playground")
                            st.info("You can edit, run, reset, or download the code below. Try your own ideas!")
                            playground_key = f"playground_{len(st.session_state.chat_history)}"
                            if f"original_code_{playground_key}" not in st.session_state:
                                st.session_state[f"original_code_{playground_key}"] = code_block
                            if f"user_code_{playground_key}" not in st.session_state:
                                st.session_state[f"user_code_{playground_key}"] = code_block
                            playground_output_key = f"playground_output_{playground_key}"
                            playground_error_key = f"playground_error_{playground_key}"
                            # --- FORM for run/reset only ---
                            with st.form(key=f"form_{playground_key}"):
                                # Always set value from session state, never from text_area return value
                                st.code(st.session_state[f"user_code_{playground_key}"], language="python")
                                col1, col2 = st.columns([1,1])
                                run_clicked = col1.form_submit_button("▶️ Run Code")
                                reset_clicked = col2.form_submit_button("🔄 Reset")
                                user_code = st.text_area(
                                    "Python code from AI (editable)",
                                    value=st.session_state[f"user_code_{playground_key}"],
                                    height=220,
                                    key=f"text_area_{playground_key}"
                                )
                                # Only update session state on submit
                                if run_clicked or reset_clicked:
                                    if reset_clicked:
                                        st.session_state[f"user_code_{playground_key}"] = st.session_state[f"original_code_{playground_key}"]
                                        st.session_state[playground_output_key] = None
                                        st.session_state[playground_error_key] = None
                                    else:
                                        st.session_state[f"user_code_{playground_key}"] = user_code
                                    # Run code and show output/errors, store output in session state to persist
                                    if run_clicked:
                                        with st.spinner("Running your code in a secure sandbox..."):
                                            try:
                                                with Sandbox(api_key=st.session_state.e2b_api_key) as code_interpreter_playground:
                                                    result = code_interpret(code_interpreter_playground, st.session_state[f"user_code_{playground_key}"])
                                                    st.session_state[playground_output_key] = result
                                                    st.session_state[playground_error_key] = None
                                            except Exception as e:
                                                st.session_state[playground_output_key] = None
                                                st.session_state[playground_error_key] = str(e)
                            # --- Download button OUTSIDE the form ---
                            st.download_button(
                                label="💾 Download Code",
                                data=st.session_state[f"user_code_{playground_key}"],
                                file_name="playground_code.py",
                                mime="text/x-python-script",
                                key=f"download_code_{playground_key}"
                            )
                            # Show output/errors if present
                            if st.session_state.get(playground_error_key):
                                st.error(f"Error running code: {st.session_state[playground_error_key]}")
                            elif st.session_state.get(playground_output_key) is not None:
                                st.success("Code executed successfully!")
                                st.markdown("**Playground Output:**")
                                result = st.session_state[playground_output_key]
                                if result:
                                    for r in result:
                                        if hasattr(r, 'png') and r.png:
                                            png_data = base64.b64decode(r.png)
                                            image = Image.open(BytesIO(png_data))
                                            st.image(image, caption="Generated Visualization", use_container_width=False)
                                        elif hasattr(r, 'figure'):
                                            fig = r.figure
                                            st.pyplot(fig)
                                        elif hasattr(r, 'show'):
                                            st.plotly_chart(r)
                                        elif isinstance(r, (pd.DataFrame, pd.Series)):
                                            st.dataframe(r)
                                        else:
                                            st.write(r)
                                else:
                                    st.info("No output or error running the code.")
                            st.markdown("---")

                        # Display results/visualizations
                        if code_results:
                            for result in code_results:
                                if hasattr(result, 'png') and result.png:  # Check if PNG data is available
                                    png_data = base64.b64decode(result.png)
                                    image = Image.open(BytesIO(png_data))
                                    st.image(image, caption="Generated Visualization", use_container_width=False)
                                elif hasattr(result, 'figure'):
                                    fig = result.figure
                                    st.pyplot(fig)
                                elif hasattr(result, 'show'):
                                    st.plotly_chart(result)
                                elif isinstance(result, (pd.DataFrame, pd.Series)):
                                    st.dataframe(result)
                                else:
                                    st.write(result)

    with tab2:
        st.header("🤖 AI Chatbot (One-to-One Conversation)")
        st.info("Chat with your data! Ask any question about your uploaded CSV. The AI will remember the conversation context, just like ChatGPT or Copilot.")
        if 'chatbot_history' not in st.session_state:
            st.session_state.chatbot_history = []  # [(role, message, timestamp)]
        import datetime
        def render_chat():
            for idx, (role, msg, ts) in enumerate(st.session_state.chatbot_history):
                if role == 'user':
                    st.markdown(f"""
<div style='background:linear-gradient(90deg,#232526,#414345);color:#fff;padding:10px;border-radius:10px;margin-bottom:5px;max-width:80%;margin-left:auto;text-align:right;box-shadow:0 2px 8px #0002;'>
<b>You</b> <span style='font-size:10px;color:#bbb;'>{ts}</span><br>{msg}
</div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
<div style='background:linear-gradient(90deg,#141e30,#243b55);color:#fff;padding:10px;border-radius:10px;margin-bottom:5px;max-width:80%;margin-right:auto;text-align:left;box-shadow:0 2px 8px #0002;'>
<b>AI</b> <span style='font-size:10px;color:#bbb;'>{ts}</span><br>{msg}
</div>""", unsafe_allow_html=True)
        dataset_path = f"./{st.session_state.uploaded_file_name}" if df is not None and st.session_state.uploaded_file_name is not None else None
        if df is not None and st.session_state.uploaded_file_name is not None:
            # --- Chatbot UI ---
            st.markdown("""
<style>
div[data-testid='stTextInput'] textarea {background:#232526;color:#fff;border-radius:8px;border:1.5px solid #ffd700;}
.stButton>button {background:#ffd700;color:#222;border-radius:8px;}
.stButton>button:hover {background:#ffe066;}
</style>
""", unsafe_allow_html=True)
            col1, col2 = st.columns([4,1])
            with col2:
                if st.button("🧹 Clear Chat", key="clear_chatbot"):
                    st.session_state.chatbot_history = []
            render_chat()
            user_input = st.text_input("Type your question for the AI:", key="chatbot_input")
            regenerate = False
            if st.button("Send", key="chatbot_send") and user_input.strip():
                regenerate = False
                last_user_input = user_input
            elif st.button("🔄 Regenerate Response", key="regenerate_chatbot") and st.session_state.chatbot_history:
                # Find last user message
                for i in range(len(st.session_state.chatbot_history)-1, -1, -1):
                    if st.session_state.chatbot_history[i][0] == 'user':
                        last_user_input = st.session_state.chatbot_history[i][1]
                        # Remove last AI response
                        if i+1 < len(st.session_state.chatbot_history) and st.session_state.chatbot_history[i+1][0]=='assistant':
                            st.session_state.chatbot_history.pop(i+1)
                        break
                regenerate = True
            else:
                last_user_input = None
            if last_user_input:
                if dataset_path is None:
                    st.error("No dataset path available. Please upload a CSV file first.")
                else:
                    # --- Enhanced Prompt Engineering ---
                    system_prompt = f"""You are a world-class, highly accurate, step-by-step data scientist AI assistant. The user has uploaded a CSV at '{dataset_path}'. Always:
- Think step by step and explain your reasoning clearly.
- Use the dataset path variable '{dataset_path}' in your code when reading the CSV file.
- If code is needed, provide a Python code block and explain the code before and after.
- If the question is ambiguous, ask clarifying questions.
- If you don't know, say so honestly.
- Always provide the most accurate, up-to-date, and relevant answer possible.
- Format your answer with clear sections, bullet points, and tables if helpful.
- Respond in a friendly, conversational, and professional tone.
"""
                    messages = [{"role": "system", "content": system_prompt}]
                    for i, (role, msg, ts) in enumerate(st.session_state.chatbot_history):
                        messages.append({"role": role, "content": msg})
                    messages.append({"role": "user", "content": last_user_input})
                    with st.spinner('AI is thinking...'):
                        client = Together(api_key=st.session_state.together_api_key)
                        response = client.chat.completions.create(
                            model=st.session_state.model_name,
                            messages=messages,
                        )
                        ai_msg = response.choices[0].message.content
                        now = datetime.datetime.now().strftime('%H:%M:%S')
                        if not regenerate:
                            st.session_state.chatbot_history.append(("user", last_user_input, now))
                        st.session_state.chatbot_history.append(("assistant", ai_msg, now))
                        st.success("AI responded!")
                        render_chat()
        else:
            st.warning("Please upload a CSV file with data in the Data Analysis tab first.")
            dataset_path = None

    with tab3:
        st.header("🧠 Auto Insights & Anomaly Detection")
        st.info("Get instant AI-powered insights, trends, and anomaly detection for your uploaded dataset. Visual explanations and suggested actions included!")
        if df is not None:
            # --- AI-Powered Insights ---
            st.subheader("AI Insights & Trends")
            if st.button("Generate Insights", key="insights_btn"):
                with st.spinner("AI is analyzing your data for insights..."):
                    # Use Together API to generate insights
                    sample_rows = df.head(100).to_csv(index=False)
                    prompt = f"""
You are a world-class data analyst. Given the following CSV data sample, provide:
- 3-5 key insights or trends (with numbers or examples)
- 1-2 possible anomalies or outliers (if any)
- 2 actionable recommendations
- Use bullet points and clear, concise language.

CSV Data Sample:
{sample_rows}
"""
                    client = Together(api_key=st.session_state.together_api_key)
                    response = client.chat.completions.create(
                        model=st.session_state.model_name,
                        messages=[{"role": "system", "content": "You are a helpful data analyst."}, {"role": "user", "content": prompt}],
                    )
                    ai_insights = response.choices[0].message.content
                    st.markdown(f"<div style='background:#232526;color:#fff;padding:16px;border-radius:10px;margin-bottom:10px;'><b>AI Insights:</b><br>{ai_insights}</div>", unsafe_allow_html=True)
            # --- Anomaly/Outlier Detection ---
            st.subheader("Automatic Anomaly & Outlier Detection")
            st.write("Detects numeric outliers using z-score and visualizes them.")
            from sklearn.preprocessing import StandardScaler
            import numpy as np
            import matplotlib.pyplot as plt
            numeric_cols = df.select_dtypes(include='number').columns
            if len(numeric_cols) == 0:
                st.warning("No numeric columns found for anomaly detection.")
            else:
                col_selected = st.selectbox("Select column for anomaly detection", numeric_cols)
                threshold = st.slider("Z-score threshold", min_value=2.0, max_value=5.0, value=3.0, step=0.1)
                data = df[col_selected].dropna().values.reshape(-1, 1)
                scaler = StandardScaler()
                z_scores = scaler.fit_transform(data)
                outlier_indices = np.where(np.abs(z_scores) > threshold)[0]
                st.write(f"Found {len(outlier_indices)} outliers in '{col_selected}'.")
                fig, ax = plt.subplots(figsize=(8,3))
                ax.plot(df[col_selected].values, label='Data', color='#1f77b4')
                ax.scatter(outlier_indices, df[col_selected].iloc[outlier_indices], color='red', label='Outliers', zorder=5)
                ax.set_title(f"Outlier Detection for {col_selected}")
                ax.legend()
                st.pyplot(fig)
                if len(outlier_indices) > 0:
                    st.markdown(f"<span style='color:#ff4b4b'><b>Action:</b> Review the highlighted outliers. Consider investigating or cleaning these data points.</span>", unsafe_allow_html=True)
        else:
            st.warning("Please upload a CSV file in the Data Analysis tab first.")

    with tab4:
        st.header("📚 Data Storytelling & Dashboards")
        if df is not None:
            subtab1, subtab2 = st.tabs(["📝 Data Storytelling & Report", "📊 Dashboard Builder"])
            # --- Data Storytelling & Report Generator ---
            with subtab1:
                st.subheader("📝 Data Storytelling & Report Generator")
                st.write("Generate a narrative report with key findings, charts, and recommendations. Export as PDF or HTML.")
                story_prompt = st.text_area(
                    "Describe the story or summary you want (or leave blank for an executive summary):",
                    "Executive summary of the uploaded dataset with key findings, trends, and recommendations."
                )
                if st.button("Generate Report", key="story_btn"):
                    with st.spinner("AI is generating your data story and report..."):
                        sample_rows = df.head(100).to_csv(index=False)
                        prompt = f"""
You are a world-class data storyteller. Given the following CSV data sample and user request, generate:
- A narrative summary (2-3 paragraphs)
- 3-5 key findings (with numbers/examples)
- 2 recommendations
- If possible, suggest and describe 1-2 charts to visualize the findings
- Use clear, engaging language

User Request: {story_prompt}

CSV Data Sample:
{sample_rows}
"""
                        client = Together(api_key=st.session_state.together_api_key)
                        response = client.chat.completions.create(
                            model=st.session_state.model_name,
                            messages=[{"role": "system", "content": "You are a helpful data storyteller."}, {"role": "user", "content": prompt}],
                        )
                        story = response.choices[0].message.content
                        st.markdown(f"<div style='background:#232526;color:#fff;padding:16px;border-radius:10px;margin-bottom:10px;'><b>AI Data Story:</b><br>{story}</div>", unsafe_allow_html=True)
                        # Export options
                        import tempfile
                        import pdfkit
                        html_report = f"<h2>AI Data Story</h2><div>{story}</div>"
                        st.download_button("Download as HTML", data=html_report, file_name="data_story.html", mime="text/html")
                        # PDF export (if pdfkit/wkhtmltopdf available)
                        try:
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_html:
                                tmp_html.write(html_report.encode("utf-8"))
                                tmp_html.flush()
                                pdf_bytes = pdfkit.from_file(tmp_html.name, False)
                                st.download_button("Download as PDF", data=pdf_bytes, file_name="data_story.pdf", mime="application/pdf")
                        except Exception as e:
                            st.info("PDF export requires pdfkit and wkhtmltopdf installed.")
            # --- AI-Powered Dashboard Builder ---
            with subtab2:
                st.subheader("📊 AI-Powered Dashboard Builder")
                st.write("Build custom dashboards by selecting chart types and columns. Export as HTML.")
                chart_types = ["Bar Chart", "Line Chart", "Scatter Plot", "Pie Chart", "Histogram"]
                chart_type = st.selectbox("Select chart type", chart_types)
                columns = df.columns.tolist()
                x_col = st.selectbox("X-axis column", columns, key="dash_x")
                y_col = st.selectbox("Y-axis column", columns, key="dash_y")
                chart = None
                import plotly.express as px
                if st.button("Generate Chart", key="dash_btn"):
                    if chart_type == "Bar Chart":
                        chart = px.bar(df, x=x_col, y=y_col, title=f"Bar Chart: {y_col} vs {x_col}")
                    elif chart_type == "Line Chart":
                        chart = px.line(df, x=x_col, y=y_col, title=f"Line Chart: {y_col} vs {x_col}")
                    elif chart_type == "Scatter Plot":
                        chart = px.scatter(df, x=x_col, y=y_col, title=f"Scatter Plot: {y_col} vs {x_col}")
                    elif chart_type == "Pie Chart":
                        chart = px.pie(df, names=x_col, values=y_col, title=f"Pie Chart: {y_col} by {x_col}")
                    elif chart_type == "Histogram":
                        chart = px.histogram(df, x=x_col, title=f"Histogram of {x_col}")
                    if chart is not None:
                        st.plotly_chart(chart, use_container_width=True)
                        plotly_html = chart.to_html(full_html=False, include_plotlyjs='cdn')
                        st.download_button(
                            label="Download Chart as HTML",
                            data=plotly_html,
                            file_name=f"dashboard_{chart_type.replace(' ','_').lower()}.html",
                            mime="text/html"
                        )
        else:
            st.warning("Please upload a CSV file in the Data Analysis tab first.")

if __name__ == "__main__":
    main()