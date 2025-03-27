import streamlit as st
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# ✅ First Streamlit Commands
st.set_page_config(page_title="ARC Master GUI", layout="wide")

# ========== Optional GPT-4V ==========
try:
    import openai
except ImportError:
    openai = None

# ========== Sidebar Data Loader ==========
st.sidebar.header("📁 Load ARC Dataset Files")
use_local_path = st.sidebar.checkbox("Use local file paths", value=True)

uploaded_files = {}
if not use_local_path:
    uploaded_files['train_challenges'] = st.sidebar.file_uploader("Upload Training Challenges", type="json")
    uploaded_files['train_solutions'] = st.sidebar.file_uploader("Upload Training Solutions", type="json")
    uploaded_files['eval_challenges'] = st.sidebar.file_uploader("Upload Evaluation Challenges", type="json")
    uploaded_files['eval_solutions'] = st.sidebar.file_uploader("Upload Evaluation Solutions", type="json")
    uploaded_files['test_challenges'] = st.sidebar.file_uploader("Upload Test Challenges", type="json")
    uploaded_files['sample_submission'] = st.sidebar.file_uploader("Upload Sample Submission", type="json")

try:
    dataset_choice = st.sidebar.selectbox("Select Dataset", ["Training", "Evaluation", "Test"])
    if use_local_path:
        paths = {
            "train_challenges": "/Users/ngoubimaximilliandiamgha/Downloads/arc-agi_training_challenges.json",
            "train_solutions": "/Users/ngoubimaximilliandiamgha/Downloads/arc-agi_training_solutions.json",
            "eval_challenges": "/Users/ngoubimaximilliandiamgha/Downloads/arc-agi_evaluation_challenges.json",
            "eval_solutions": "/Users/ngoubimaximilliandiamgha/Downloads/arc-agi_evaluation_solutions.json",
            "test_challenges": "/Users/ngoubimaximilliandiamgha/Downloads/arc-agi_test_challenges.json",
            "sample_submission": "/Users/ngoubimaximilliandiamgha/Downloads/sample_submission.json",
        }
        if dataset_choice == "Training":
            challenges_data = json.load(open(paths['train_challenges']))
            solutions_data = json.load(open(paths['train_solutions']))
        elif dataset_choice == "Evaluation":
            challenges_data = json.load(open(paths['eval_challenges']))
            solutions_data = json.load(open(paths['eval_solutions']))
        else:
            challenges_data = json.load(open(paths['test_challenges']))
            solutions_data = json.load(open(paths['sample_submission']))
    else:
        key_map = {
            "Training": ('train_challenges', 'train_solutions'),
            "Evaluation": ('eval_challenges', 'eval_solutions'),
            "Test": ('test_challenges', 'sample_submission')
        }
        k1, k2 = key_map[dataset_choice]
        challenges_data = json.load(uploaded_files[k1])
        solutions_data = json.load(uploaded_files[k2])
    st.sidebar.success("✅ Data loaded successfully")
except Exception as e:
    st.sidebar.error(f"❌ Failed to load data: {e}")
    st.stop()

# ========== Visualization ==========
def plot_grid(grid, title=""):
    try:
        arr = np.array(grid)
        if arr.dtype == object:
            arr = arr.astype(int)
        fig, ax = plt.subplots()
        ax.imshow(arr, cmap='tab10', vmin=0, vmax=9)
        ax.set_title(title)
        ax.axis('off')
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Visualization failed: {e}")

# ========== Dataset Insights ==========
with st.expander("📊 Dataset Stats: Grid Size & Color Frequency"):
    grid_shapes = []
    color_counter = Counter()
    for task in challenges_data.values():
        for pair in task['train'] + task.get('test', []):
            g = pair['input']
            grid_shapes.append(f"{len(g)}x{len(g[0])}")
            color_counter.update(np.array(g).flatten())
    st.subheader("Grid Size Distribution")
    shapes, counts = zip(*Counter(grid_shapes).items())
    fig1, ax1 = plt.subplots()
    ax1.bar(shapes, counts)
    ax1.set_ylabel("Count")
    ax1.set_title("Grid Sizes")
    st.pyplot(fig1)
    st.subheader("Color Frequency (0–9)")
    colors = list(range(10))
    freqs = [color_counter[i] for i in colors]
    fig2, ax2 = plt.subplots()
    ax2.bar(colors, freqs, color=plt.cm.tab10.colors)
    st.pyplot(fig2)

# ========== Real Reasoning Models ==========
class RuleInferencer:
    def infer(self, input_grid, output_grid):
        rules = []
        if np.array_equal(np.flip(input_grid, axis=1), output_grid):
            rules.append("horizontal_flip")
        if np.array_equal(np.flip(input_grid, axis=0), output_grid):
            rules.append("vertical_flip")
        for k in range(1, 4):
            if np.array_equal(np.rot90(input_grid, k=k), output_grid):
                rules.append(f"rotation_{k * 90}")
        return rules or ["unknown"]

class RuleBasedReasoner:
    def solve_task(self, task):
        preds, ri = [], RuleInferencer()
        for test in task['test']:
            rules = []
            for pair in task['train']:
                rules = ri.infer(np.array(pair['input']), np.array(pair['output']))
                if rules[0] != "unknown": break
            grid = np.array(test['input'])
            for rule in rules:
                if rule == "horizontal_flip":
                    grid = np.flip(grid, axis=1)
                elif rule == "vertical_flip":
                    grid = np.flip(grid, axis=0)
                elif rule.startswith("rotation"):
                    grid = np.rot90(grid, k=int(rule.split('_')[1]) // 90)
            preds.append(grid.tolist())
        return preds, ["Rule-Based reasoning"] * len(preds)

class XGBoostModel:
    def __init__(self):
        self.model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
        self.label_encoder = LabelEncoder()

    def preprocess(self, grid):
        arr = np.array(grid)
        arr = arr / 9.0  # Normalize the values between 0 and 1
        return arr.flatten().reshape(1, -1)

    def solve_task(self, task):
        # First, fit the model on the training data
        X_train = []
        y_train = []
        for task_data in task['train']:
            X_train.append(self.preprocess(task_data['input']))
            y_train.append(task_data['output'])
        # Encode the target labels to ensure they're in a continuous range
        y_train = np.array(y_train)
        y_train_encoded = self.label_encoder.fit_transform(y_train.flatten())
        X_train = np.vstack(X_train)
        # Fit the model
        self.model.fit(X_train, y_train_encoded)
        # Make predictions on the test data
        preds, exps = [], []
        for test in task['test']:
            inp = self.preprocess(test['input'])
            pred = self.model.predict(inp)
            pred = self.label_encoder.inverse_transform(pred)  # Decode the predicted labels
            preds.append(pred.reshape(test['input'].shape))  # Reshape to match original grid shape
            exps.append("XGBoost model processed the grid")
        return preds, exps

class SVMModel:
    def __init__(self):
        self.model = SVC(kernel='linear')

    def preprocess(self, grid):
        arr = np.array(grid)
        arr = arr / 9.0  # Normalize the values between 0 and 1
        return arr.flatten()

    def solve_task(self, task):
        # First, fit the model on the training data
        X_train = []
        y_train = []
        for task_data in task['train']:
            X_train.append(self.preprocess(task_data['input']))
            y_train.append(task_data['output'])

        # Convert to proper format for fitting
        X_train = np.vstack(X_train)  # Stack into a 2D array
        y_train = np.array(y_train).flatten()  # Flatten labels into 1D array

        # Fit the model
        self.model.fit(X_train, y_train)

        # Make predictions on the test data
        preds, exps = [], []
        for test in task['test']:
            inp = self.preprocess(test['input'])
            inp = inp.reshape(1, -1)  # Ensure it's 2D for prediction
            pred = self.model.predict(inp)
            preds.append(pred.reshape(test['input'].shape))  # Reshape to match original grid shape
            exps.append("SVM model processed the grid")
        return preds, exps

REASONING_ENGINES = {
    "Rule-Based": RuleBasedReasoner().solve_task,
    "XGBoost Classifier": XGBoostModel().solve_task,
    "SVM Model": SVMModel().solve_task,
}

# ========== Main Interface ==========
st.title("🧠 ARC Master GUI — Real Reasoning Models")

task_id = st.selectbox("Select Task ID", list(challenges_data.keys()))
task = challenges_data[task_id]
solution = solutions_data.get(task_id, [])
model_choice = st.selectbox("Choose Reasoning Model", list(REASONING_ENGINES.keys()))
predict_fn = REASONING_ENGINES[model_choice]

with st.spinner("Running model..."):
    predictions, explanations = predict_fn(task)

st.subheader("🔧 Training Examples")
for i, pair in enumerate(task['train']):
    col1, col2 = st.columns(2)
    with col1:
        plot_grid(pair['input'], f"Train Input {i + 1}")
    with col2:
        plot_grid(pair['output'], f"Train Output {i + 1}")

st.subheader("🧪 Test Predictions")
for i, test in enumerate(task['test']):
    col1, col2, col3 = st.columns(3)
    with col1:
        plot_grid(test['input'], f"Test Input {i + 1}")
    with col2:
        plot_grid(predictions[i], f"Predicted Output {i + 1}")
    with col3:
        plot_grid(solution[i], f"Ground Truth {i + 1}")
    st.caption(f"✅ Match: {np.array_equal(np.array(predictions[i]), np.array(solution[i]))}")
    with st.expander(f"📖 Explanation for Test {i + 1}"):
        st.code(explanations[i])
