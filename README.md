<img width="1024" height="1024" alt="AI" src="https://github.com/user-attachments/assets/cdde8142-8f62-48f4-8eb3-ecc0726ff381" />

Try OUR APP:-- 

<img width="1916" height="1022" alt="page0" src="https://github.com/user-attachments/assets/5d7ea735-0ce1-4ad0-a7f8-4e4b7c5b59e1" />


https://github.com/user-attachments/assets/8595554e-6fd6-4acf-9da6-b8cc3aef611c


---

# 📊 Data Visualization Saathi AI Agent

### *"Your Smartest Data Companion for Insightful Visualizations"*

#### 🚀 Powered by TechSeva Solutions

---

## 🛠️ What I Built Today

Today, I built something truly transformative for anyone working with data:

**👉 Data Visualization Saathi AI Agent** — a beautifully branded, AI-powered data analytics and visualization assistant built using cutting-edge technologies, modern UI/UX, and advanced agentic architecture.

Whether you're a data scientist, analyst, or beginner, this app makes it effortless to explore, analyze, and visualize your data—**with just conversations.**

---

## 🔍 App Title:

**📊 Data Visualization Saathi AI Agent**

## 🏷️ Tagline:

*"Your Smartest Data Companion for Insightful Visualizations"*

## 💼 Powered By:

**TechSeva Solutions** (with custom animated design & dark neon UI)

---

## 🧠 Core Idea

**A full-stack, conversational AI-powered data analysis tool** that enables:

* Seamless data upload
* Smart insight generation
* Natural-language Q\&A
* Automated code generation
* Dashboards and storytelling
  —all with an engaging and branded interface.

---

## 📌 Core Features (Deep Dive)

### 🔹 1. Data Upload & Management

* Upload any **CSV file** to get started
* **Defensive error handling** for invalid/empty files
* Uses **Session State** to manage files, user settings, and conversations

---

### 🔹 2. Data Analysis Tab

#### 📂 Dataset Preview

* View the **full dataset** or just the **first 5 rows**

#### 🧠 Insight Suggestions (Auto-generated)

* Detects:

  * Columns with **missing values**
  * **Categorical** features
  * **Correlated** numeric pairs
  * Outliers and their possible reasons

#### 📊 Automatic Data Profiling

* **Descriptive stats** and metadata
* Interactive charts via **Plotly** and **Altair**
* Export charts to **HTML/JSON**

#### 🤖 AI Data Q\&A

* Ask questions like:

  * "What’s the correlation between price and sales?"
  * "Show me average revenue by category"
* AI replies with **code + explanation**
* View output tables, charts, and summaries

#### 🧪 Code Playground

* AI-generated code is editable, executable, and downloadable
* Outputs tables, images, and interactive visualizations

#### 📜 Conversational History

* Full history of Q\&A displayed for reference

---

### 🔹 3. AI Chatbot Tab

* 🔁 One-to-one **chat** with your dataset
* 💬 **Memory**: Maintains conversation context
* 🔄 **Regenerate** previous answer
* ❌ Clear chat history anytime
* 💅 Modern chat UI with smooth animations

---

### 🔹 4. Auto Insights & Anomaly Detection Tab

#### 📌 AI-Generated Insights

* App summarizes:

  * 3–5 key data insights
  * Anomalies
  * Actionable recommendations

#### 📉 Anomaly Detection

* Choose a **numeric column**
* Set a **z-score threshold**
* Visualize outliers with context + suggestions

---

### 🔹 5. Data Storytelling & Dashboards Tab

#### 📖 Narrative Report Generator

* Auto-generated:

  * Summary
  * Key Findings
  * Recommendations
  * Chart suggestions
* Export to **PDF or HTML**

#### 📊 Dashboard Builder

* Choose chart type:
  **Bar, Line, Pie, Histogram, Scatter**
* Select columns → generate interactive dashboards
* Download charts in HTML format

---

## 🎨 UX, UI & Dev Experience Highlights

| Feature                      | Description                                                     |
| ---------------------------- | --------------------------------------------------------------- |
| 🖌️ **Modern Animated UI**   | Neon-themed dark interface, glowing effects, interactive layout |
| 🎥 **Sidebar Branding**      | Animated logos, developer name, and optional profile photo      |
| 🔐 **API Key Handling**      | Uses `.env` for Together.ai and E2B sandbox keys                |
| 🧠 **Model Selection**       | Choose among LLMs for customized performance                    |
| 🔄 **State Management**      | Tracks files, settings, code, and chat in Streamlit sessions    |
| 🧱 **Robust Error Handling** | Catch upload, code, API, and runtime errors gracefully          |
| 🧑‍💻 **Developer Identity** | Fully branded under **TechSeva Solutions** with polished UI     |

---

## 🔁 User Flow

```text
1. Upload your CSV data
2. Explore it using data preview and AI Q&A
3. Analyze missing values, outliers, and key columns
4. Ask specific questions — get code + results
5. Generate reports and dashboards with AI
6. Save or download charts, code, and reports
```

---

## 🔧 Technologies Used

| Tech              | Purpose                              |
| ----------------- | ------------------------------------ |
| Streamlit         | Web App Frontend                     |
| Python            | Backend logic & AI orchestration     |
| Plotly/Altair     | Interactive Visualizations           |
| Together.ai / E2B | LLM access + code sandbox            |
| dotenv            | API key management                   |
| PDFKit            | PDF export functionality             |
| Custom CSS        | Animated, branded UI styling         |
| Session State     | Persistent user interaction tracking |

---

## 🧾 Summary

**📊 Data Visualization Saathi AI Agent** is a smart, modern, AI-first tool that helps users:

* Understand their data
* Ask questions naturally
* Visualize trends and outliers
* Tell compelling data stories
* Generate ready-to-use dashboards

All packed inside a gorgeous, branded interface that feels **intelligent, intuitive, and immersive.**


---


---

### ✅ **Project: Data Visualization Saathi AI Agent**

**LangGraph-style Agent Workflow**
🧠 **Powered by OpenAI (or Together.ai) & TechSeva Solutions**

---

## 🧩 **LangGraph Workflow Overview**

```
User Uploads CSV / Enters Prompt
        │
        ▼
📁 File Handler Node ──▶ Checks Validity / Stores DataFrame in Memory
        │
        ▼
🧠 Data Profiler Node
   └─→ Describes: Rows, Columns, Nulls, Stats
   └─→ Detects: Categorical, Numeric, Outliers
        │
        ▼
🤖 Question Analyzer Node
   └─→ Classifies user intent:
        - Ask a data question?
        - Request insight/anomaly?
        - Ask for storytelling/dashboard?
        - Code-related request?
        │
        ▼
🎯 Decision Router Node (Conditional Split)
 ┌────────────────────────┬─────────────────────────┬──────────────────────┐
 │                        │                         │                      │
 ▼                        ▼                         ▼                      ▼
📊 Data Q&A Node       📈 Auto Insight Node      🛠️ Code Sandbox Node   📖 Storytelling Node
AI answers user’s Q   AI generates 3–5         AI creates, edits,     AI writes summary,
via explanation +     insights, anomaly        runs or debugs code    builds narrative
generated code        detection, suggestions   → Plots, Tables        exports to PDF/HTML

         ▼                          ▼                         ▼                         ▼
   🗂️ Q&A History Node      📉 Anomaly Visualizer      🧪 Code Output        📄 Export Report Node
       (tracks convos)        (plotly or altair)       (Live Preview)        (Markdown / PDF / HTML)

                                ▼
                   📤 Output to User Interface
```

---

## 🧱 **Key Nodes in the Workflow**

| Node                   | Function                                               |
| ---------------------- | ------------------------------------------------------ |
| `FileHandlerNode`      | Upload & validate CSV file, parse into DataFrame       |
| `DataProfilerNode`     | Automatic EDA – stats, nulls, column types             |
| `QuestionAnalyzerNode` | Classifies prompt intent (chat vs task vs code)        |
| `DecisionRouterNode`   | Directs to Q\&A, Insight, Code, or Storytelling        |
| `DataQnANode`          | Answers based on DataFrame + generates plots/code      |
| `AutoInsightNode`      | Finds patterns, correlations, and data anomalies       |
| `CodeSandboxNode`      | Displays runnable/AI-generated code with edit + output |
| `StorytellingNode`     | Creates narratives, findings, and recommendations      |
| `QnAHistoryNode`       | Stores previous questions and answers                  |
| `AnomalyVisualizer`    | Plots anomalies with z-score thresholds                |
| `ExportReportNode`     | Allows exporting output (charts/text) as PDF/HTML/MD   |

---

## 🔁 Looping Feedback

Each major node allows users to:

* Ask follow-up questions
* Regenerate results
* Edit and rerun code
* Return to previous tab (via session memory)

---


## 🎨 UI Mapping to Nodes

| UI Component         | Connected Node         |
| -------------------- | ---------------------- |
| Sidebar              | Memory, Model Selector |
| File Uploader        | FileHandlerNode        |
| Data Preview Tabs    | DataProfilerNode       |
| Q\&A Chatbox         | DataQnANode            |
| Code Editor + Output | CodeSandboxNode        |
| Insight Dashboard    | AutoInsightNode        |
| Narrative Generator  | StorytellingNode       |
| Download Buttons     | ExportReportNode       |

---

## 🧠 LangGraph Summary

**Agent Type:** Agent Executor w/ Decision Router
**Graph Type:** Conditional DAG with conversational memory
**Memory:** Session-based contextual + code+chat history
**Execution Flow:** Prompt → Router → Branch → Output → Memory → Backtrack/Fork → Save




## 💻 Run Locally

```bash
git clone https://github.com/abhishekkumar62000/Data-Visualization-Saathi-AI-Agent.git
cd data-viz-saathi-agent
pip install -r requirements.txt
streamlit run app.py
```

---

## 🙌 Contribute / Feedback

Have ideas, found a bug, or want to collaborate?
Feel free to open an issue or pull request.

**Built with ❤️ by TechSeva Solutions*  Developer:- Abhishek Kumar*

---
