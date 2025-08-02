import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import os
import json
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# Page configuration
st.set_page_config(page_title="Disease Prediction System", layout="wide")
sns.set_style("whitegrid")


# ------------------------------
# Style and background settings
# ------------------------------
def set_bg_image(img_path):
    """Set page background image"""
    if not os.path.exists(img_path):
        st.warning(f"Background image {img_path} not found. Using default background.")
        return
    
    with open(img_path, "rb") as img_file:
        img_b64 = base64.b64encode(img_file.read()).decode()
    
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/jpg;base64,{img_b64});
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        .stTextInput > div > div > input, .stNumberInput > div > div > input {{
            background-color: #ffffff;
            border-radius: 4px;
        }}
        .main-title {{
            color: #2c3e50;
            text-align: center;
            margin-bottom: 2rem;
        }}
        .section-title {{
            color: #34495e;
            border-left: 4px solid #e74c3c;
            padding-left: 10px;
            margin-top: 1.5rem;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


def set_login_bg(img_path):
    """Set login page background"""
    if not os.path.exists(img_path):
        st.warning(f"Login background image {img_path} not found. Using default background.")
        return
    
    with open(img_path, "rb") as img_file:
        img_b64 = base64.b64encode(img_file.read()).decode()
    
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/jpg;base64,{img_b64});
            background-size: cover;
            background-position: center;
        }}
        .login-card {{
            background-color: rgba(255, 255, 255, 0.9);
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


# ------------------------------
# Data file management
# ------------------------------
def init_file(file_path, default_content):
    """Initialize file if it doesn't exist"""
    if not os.path.exists(file_path):
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(default_content, f, ensure_ascii=False)


def load_json(file_path, default=None):
    """Read content from JSON file with safe default"""
    if default is None:
        default = {}
    init_file(file_path, default)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, Exception) as e:
        st.warning(f"Error reading {file_path}: {str(e)}. Using default data.")
        return default


def save_json(file_path, data):
    """Save data to JSON file with error handling"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        return True
    except Exception as e:
        st.error(f"Error saving to {file_path}: {str(e)}")
        return False


# ------------------------------
# User authentication
# ------------------------------
def verify_user(username, password):
    """Verify user login credentials"""
    users = load_json('users.json')
    user_info = users.get(username)
    if user_info and user_info['password'] == password:
        return True, username == '1neo9'  # (login success, is admin)
    return False, False


def add_new_user(username, password, gender, age, nickname):
    """Register a new user"""
    users = load_json('users.json')
    if username in users:
        return False  # Username already exists
    users[username] = {
        'password': password,
        'gender': gender,
        'age': age,
        'nickname': nickname,
        'is_admin': False
    }
    return save_json('users.json', users)


# ------------------------------
# Data processing and model training
# ------------------------------
@st.cache_data
def load_health_data():
    """Load and clean health data, with fallback to sample data if file not found"""
    if not os.path.exists('heart_0531.xlsx'):
        st.warning("Health data file not found. Using sample data for demonstration.")
        
        # Create sample data
        np.random.seed(42)
        data = {
            'age': np.random.randint(30, 80, 300),
            'sex': np.random.randint(0, 2, 300),
            'trestbps': np.random.randint(90, 160, 300),
            'chol': np.random.randint(120, 300, 300),
            'fbs': np.random.randint(0, 2, 300),
            'thalach': np.random.randint(100, 200, 300),
            'exang': np.random.randint(0, 2, 300),
            'thal': np.random.randint(0, 3, 300),
            'target': np.random.randint(0, 2, 300)
        }
        return pd.DataFrame(data)
    
    try:
        raw_data = pd.read_excel('health_data.xlsx')
        clean_data = raw_data.dropna()
        
        # Remove outliers
        def drop_outliers(df):
            q1 = df.quantile(0.25)
            q3 = df.quantile(0.75)
            iqr = q3 - q1
            return df[~((df < (q1 - 1.5 * iqr)) | (df > (q3 + 1.5 * iqr))).any(axis=1)]
        
        return drop_outliers(clean_data)
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        # Create fallback sample data
        np.random.seed(42)
        data = {
            'age': np.random.randint(30, 80, 300),
            'sex': np.random.randint(0, 2, 300),
            'trestbps': np.random.randint(90, 160, 300),
            'chol': np.random.randint(120, 300, 300),
            'fbs': np.random.randint(0, 2, 300),
            'thalach': np.random.randint(100, 200, 300),
            'exang': np.random.randint(0, 2, 300),
            'thal': np.random.randint(0, 3, 300),
            'target': np.random.randint(0, 2, 300)
        }
        return pd.DataFrame(data)


@st.cache_resource
def build_model(health_data):
    """Train random forest classification model"""
    # Features and target variable
    features = ['age', 'sex', 'trestbps', 'chol', 'fbs', 'thalach', 'exang', 'thal']
    
    # Check if all required features exist
    missing_features = [f for f in features if f not in health_data.columns]
    if missing_features:
        st.warning(f"Missing features in data: {', '.join(missing_features)}. Using available features.")
        features = [f for f in features if f in health_data.columns]
    
    # Ensure target exists
    if 'target' not in health_data.columns:
        st.warning("Target column not found. Creating dummy target for demonstration.")
        health_data['target'] = np.random.randint(0, 2, len(health_data))
    
    X = health_data[features]
    y = health_data['target']
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42
    )
    
    # Grid search for optimal model
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        verbose=2
    )
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_, X_test, y_test


# ------------------------------
# Announcement management
# ------------------------------
def get_all_announcements():
    """Get all announcements"""
    return load_json('announcements.json')


def add_announcement(title, content, author):
    """Add new announcement"""
    announcements = get_all_announcements()
    ann_id = f"ANN_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    announcements[ann_id] = {
        'title': title,
        'content': content,
        'author': author,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    return save_json('announcements.json', announcements)


def update_announcement(ann_id, new_title, new_content):
    """Update announcement content"""
    announcements = get_all_announcements()
    if ann_id in announcements:
        announcements[ann_id]['title'] = new_title
        announcements[ann_id]['content'] = new_content
        return save_json('announcements.json', announcements)
    return False


def remove_announcement(ann_id):
    """Delete announcement"""
    announcements = get_all_announcements()
    if ann_id in announcements:
        del announcements[ann_id]
        return save_json('announcements.json', announcements)
    return False


# ------------------------------
# Prediction history management
# ------------------------------
def save_prediction_record(username, probability, input_data):
    """Save prediction record with proper initialization"""
    # Ensure directories exist
    if not os.path.exists('user_records'):
        os.makedirs('user_records')
    
    # Save prediction result with proper initialization
    pred_file = f'user_records/{username}_risk_records.json'
    pred_records = load_json(pred_file, default=[])  # Ensure default is empty list
    
    # Add new record
    new_record = {
        'probability': probability,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    pred_records.append(new_record)
    
    # Save input data
    input_file = f'user_records/{username}_input_data.json'
    input_records = load_json(input_file, default=[])  # Ensure default is empty list
    
    input_data_with_time = input_data.copy()
    input_data_with_time['timestamp'] = new_record['timestamp']
    input_records.append(input_data_with_time)
    
    # Save both files
    pred_success = save_json(pred_file, pred_records)
    input_success = save_json(input_file, input_records)
    
    return pred_success and input_success


def get_prediction_history(username):
    """Get prediction history with proper error handling"""
    pred_file = f'user_records/{username}_risk_records.json'
    return load_json(pred_file, default=[])


def get_input_history(username):
    """Get input history with proper error handling"""
    input_file = f'user_records/{username}_input_data.json'
    return load_json(input_file, default=[])


# ------------------------------
# Page rendering - Authentication
# ------------------------------
def show_login_page():
    """Display login page"""
    with st.container():
        st.markdown('<div class="login-card">', unsafe_allow_html=True)
        st.subheader("User Login")
        
        username = st.text_input("Enter username")
        password = st.text_input("Enter password", type="password")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Login", use_container_width=True):
                success, is_admin = verify_user(username, password)
                if success:
                    st.session_state.update({
                        'logged_in': True,
                        'current_user': username,
                        'is_admin': is_admin,
                        'page': 'Data Analysis'
                    })
                    st.rerun()
                else:
                    st.error("Incorrect username or password")
        
        with col2:
            if st.button("Create New Account", use_container_width=True):
                st.session_state['page'] = 'User Registration'
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


def show_register_page():
    """Display registration page"""
    with st.container():
        st.markdown('<div class="login-card">', unsafe_allow_html=True)
        st.subheader("User Registration")
        
        new_user = st.text_input("Set username")
        new_pwd = st.text_input("Set password (at least 6 characters)", type="password")
        confirm_pwd = st.text_input("Confirm password", type="password")
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", min_value=0, max_value=120, value=18)
        nickname = st.text_input("Set nickname (required)")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Complete Registration", use_container_width=True):
                if new_pwd != confirm_pwd:
                    st.error("Passwords do not match")
                elif len(new_pwd) < 6:
                    st.warning("Password must be at least 6 characters")
                elif not nickname.strip():
                    st.warning("Nickname cannot be empty")
                else:
                    if add_new_user(new_user, new_pwd, gender, age, nickname):
                        st.success("Registration successful, redirecting to login page")
                        st.session_state['page'] = 'User Login'
                        st.rerun()
                    else:
                        st.warning("Username already exists, please choose another")
        
        with col2:
            if st.button("Back to Login", use_container_width=True):
                st.session_state['page'] = 'User Login'
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


# ------------------------------
# Page rendering - Function pages
# ------------------------------
def show_dashboard(health_data, model, X_test, y_test):
    """Display data analysis dashboard"""
    st.markdown('<h2 class="main-title">Health Data Analysis</h2>', unsafe_allow_html=True)
    
    # Continuous variable distribution
    st.markdown('<h3 class="section-title">1. Continuous Metrics Distribution</h3>', unsafe_allow_html=True)
    cont_vars = ['age', 'trestbps', 'chol', 'thalach']
    cont_vars = [v for v in cont_vars if v in health_data.columns]
    cols = st.columns(2)
    for i, var in enumerate(cont_vars):
        with cols[i % 2]:
            fig, ax = plt.subplots(figsize=(5, 3))
            sns.histplot(health_data[var], kde=True, ax=ax, color='#e74c3c')
            ax.set_title(f'Distribution of {var}')
            ax.set_xlabel(var)
            ax.set_ylabel('Frequency')
            st.pyplot(fig)
    
    # Categorical variable distribution
    st.markdown('<h3 class="section-title">2. Categorical Metrics Distribution</h3>', unsafe_allow_html=True)
    cat_vars = ['sex', 'fbs', 'exang', 'thal']
    cat_vars = [v for v in cat_vars if v in health_data.columns]
    cols = st.columns(2)
    for i, var in enumerate(cat_vars):
        with cols[i % 2]:
            fig, ax = plt.subplots(figsize=(5, 3))
            counts = health_data[var].value_counts()
            ax.pie(counts, labels=counts.index, autopct='%1.1f%%', 
                   colors=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'])
            ax.set_title(f'Distribution of {var}')
            st.pyplot(fig)
    
    # Condition status vs metrics
    if 'target' in health_data.columns:
        st.markdown('<h3 class="section-title">3. Condition Status vs Metrics</h3>', unsafe_allow_html=True)
        cols = st.columns(2)
        for i, var in enumerate(cont_vars):
            with cols[i % 2]:
                fig, ax = plt.subplots(figsize=(5, 3))
                sns.boxplot(x='target', y=var, data=health_data, ax=ax, palette=['#2ecc71', '#e74c3c'])
                ax.set_xlabel('Condition Status')
                ax.set_ylabel(var)
                st.pyplot(fig)
    
    # Correlation analysis
    st.markdown('<h3 class="section-title">4. Metrics Correlation Analysis</h3>', unsafe_allow_html=True)
    if cont_vars:
        corr_data = health_data[cont_vars].corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_data, annot=True, cmap='viridis', fmt='.2f', ax=ax)
        ax.set_title('Correlation Matrix')
        st.pyplot(fig)
    else:
        st.info("No continuous variables available for correlation analysis")
    
    # Model performance
    st.markdown('<h3 class="section-title">5. Model Performance Evaluation</h3>', unsafe_allow_html=True)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Performance metrics
    report = classification_report(y_test, y_pred, output_dict=True)
    st.write(f"Accuracy: {report['accuracy']:.4f}")
    if '1' in report:
        st.write(f"Condition Detection Rate: {report['1']['recall']:.4f}")
        st.write(f"Prediction Precision: {report['1']['precision']:.4f}")
    else:
        st.write("Insufficient data for complete performance metrics")
    
    # Confusion matrix
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='magma', ax=ax)
    ax.set_xlabel('Predicted Result')
    ax.set_ylabel('Actual Result')
    st.pyplot(fig)
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fpr, tpr, color='#3498db', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend()
    st.pyplot(fig)


def show_prediction(model):
    """Display disease prediction page with fixed save functionality"""
    st.markdown('<h2 class="main-title">Health Risk Assessment</h2>', unsafe_allow_html=True)
    
    with st.form("pred_form"):
        st.write("Please enter the following health metrics for risk assessment:")
        col1, col2 = st.columns(2)
        
        # Input fields
        input_vals = {}
        fields = [
            ('age', 'Age (years)', 50),
            ('sex', 'Gender (0=Female, 1=Male)', 0),
            ('trestbps', 'Resting Blood Pressure (mm Hg)', 120),
            ('chol', 'Serum Cholesterol (mg/dl)', 200),
            ('fbs', 'Fasting Blood Sugar >120mg/dl (0=No, 1=Yes)', 0),
            ('thalach', 'Maximum Heart Rate', 150),
            ('exang', 'Exercise Induced Angina (0=No, 1=Yes)', 0),
            ('thal', 'Thalassemia Type (0=Normal, 1=Fixed, 2=Reversible)', 0)
        ]
        
        for i, (key, label, default) in enumerate(fields):
            with col1 if i % 2 == 0 else col2:
                input_vals[key] = st.number_input(label, value=default, step=1)
        
        submit_btn = st.form_submit_button("Start Assessment", use_container_width=True)
    
    if submit_btn:
        try:
            # Generate prediction result
            input_df = pd.DataFrame([input_vals])
            risk_prob = model.predict_proba(input_df)[0][1] * 100
            st.success(f"Health Risk Assessment Result: **{risk_prob:.2f}%**")
            
            # Save records using the new dedicated function
            username = st.session_state['current_user']
            save_success = save_prediction_record(
                username, 
                f'{risk_prob:.2f}', 
                input_vals
            )
            
            if not save_success:
                st.warning("Could not save assessment history. Functionality is not affected.")
                
        except Exception as e:
            st.error(f"An error occurred during assessment: {str(e)}")


def show_user_profile():
    """Display user profile page"""
    st.markdown('<h2 class="main-title">User Profile</h2>', unsafe_allow_html=True)
    username = st.session_state['current_user']
    users = load_json('users.json')
    user_info = users.get(username)
    
    if not user_info:
        st.error("User information not found, please log in again")
        return
    
    # Basic information
    st.markdown('<h3 class="section-title">Basic Information</h3>', unsafe_allow_html=True)
    st.write(f"Username: {username}")
    st.write(f"Nickname: {user_info['nickname']}")
    st.write(f"Gender: {user_info['gender']}")
    st.write(f"Age: {user_info['age']}")
    
    # Change nickname
    st.markdown('<h3 class="section-title">Update Nickname</h3>', unsafe_allow_html=True)
    new_nick = st.text_input("New Nickname", value=user_info['nickname'])
    if st.button("Save Nickname"):
        users[username]['nickname'] = new_nick
        if save_json('users.json', users):
            st.success("Nickname updated successfully")
        else:
            st.error("Failed to update nickname")
    
    # Change password
    st.markdown('<h3 class="section-title">Update Password</h3>', unsafe_allow_html=True)
    old_pwd = st.text_input("Current Password", type="password")
    new_pwd = st.text_input("New Password", type="password")
    confirm_pwd = st.text_input("Confirm New Password", type="password")
    
    if st.button("Update Password"):
        if old_pwd != user_info['password']:
            st.error("Current password is incorrect")
        elif new_pwd != confirm_pwd:
            st.error("New passwords do not match")
        elif not new_pwd:
            st.error("New password cannot be empty")
        else:
            users[username]['password'] = new_pwd
            if save_json('users.json', users):
                st.success("Password updated successfully")
            else:
                st.error("Failed to update password")
    
    # Admin messages
    msg = load_json('messages.json', default={}).get(username, "")
    if msg:
        st.markdown('<h3 class="section-title">Administrator Message</h3>', unsafe_allow_html=True)
        st.info(msg)
    
    # History records
    st.markdown('<h3 class="section-title">Assessment History</h3>', unsafe_allow_html=True)
    pred_records = get_prediction_history(username)
    
    if not pred_records:
        st.info("No assessment records yet")
    else:
        input_records = get_input_history(username)
        for rec in reversed(pred_records):
            with st.expander(f"Assessment Time: {rec['timestamp']} | Risk Value: {rec['probability']}%"):
                try:
                    # Find corresponding input data
                    for data in reversed(input_records):
                        if data.get('timestamp') == rec['timestamp']:
                            st.json({k: v for k, v in data.items() if k != 'timestamp'})
                            break
                except Exception as e:
                    st.warning(f"Error loading input data: {str(e)}")
    
    # Logout
    st.markdown('<h3 class="section-title">Account Operations</h3>', unsafe_allow_html=True)
    if st.button("Logout"):
        st.session_state['logged_in'] = False
        st.session_state['page'] = 'User Login'
        st.rerun()


def show_announcement_management():
    """Display admin announcement management page"""
    st.markdown('<h2 class="main-title">Announcement Management</h2>', unsafe_allow_html=True)
    
    # Publish new announcement
    st.markdown('<h3 class="section-title">Publish New Announcement</h3>', unsafe_allow_html=True)
    with st.form("new_ann_form"):
        title = st.text_input("Announcement Title")
        content = st.text_area("Announcement Content")
        if st.form_submit_button("Publish Announcement"):
            if not title or not content:
                st.warning("Title and content cannot be empty")
            else:
                if add_announcement(title, content, st.session_state['current_user']):
                    st.success("Announcement published successfully")
                    st.rerun()
                else:
                    st.error("Failed to publish announcement")
    
    # Search and filter
    search_key = st.text_input("Search Announcement Title")
    announcements = get_all_announcements()
    filtered_anns = {k: v for k, v in announcements.items() 
                    if search_key.lower() in v['title'].lower()}
    
    # Announcement list
    st.markdown('<h3 class="section-title">Announcement List</h3>', unsafe_allow_html=True)
    if not filtered_anns:
        st.info("No announcements available")
    else:
        for ann_id, ann in reversed(filtered_anns.items()):
            with st.expander(f"{ann['title']} ({ann['timestamp']})"):
                st.write(f"Content: {ann['content']}")
                st.write(f"Author: {ann['author']}")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Delete", key=f"del_{ann_id}"):
                        if remove_announcement(ann_id):
                            st.success("Announcement deleted successfully")
                            st.rerun()
                        else:
                            st.error("Failed to delete announcement")
                with col2:
                    if st.button("Edit", key=f"edit_{ann_id}"):
                        st.session_state['edit_ann_id'] = ann_id
                        st.rerun()
    
    # Edit announcement
    if 'edit_ann_id' in st.session_state:
        ann_id = st.session_state['edit_ann_id']
        announcements = get_all_announcements()
        ann = announcements.get(ann_id)
        if ann:
            st.markdown('<h3 class="section-title">Edit Announcement</h3>', unsafe_allow_html=True)
            new_title = st.text_input("Announcement Title", value=ann['title'])
            new_content = st.text_area("Announcement Content", value=ann['content'])
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Save Changes"):
                    if update_announcement(ann_id, new_title, new_content):
                        del st.session_state['edit_ann_id']
                        st.success("Announcement updated successfully")
                        st.rerun()
                    else:
                        st.error("Failed to update announcement")
            with col2:
                if st.button("Cancel Editing"):
                    del st.session_state['edit_ann_id']
                    st.rerun()


def show_public_announcements():
    """Display public announcements page"""
    st.markdown('<h2 class="main-title">Announcements & Notices</h2>', unsafe_allow_html=True)
    
    search_key = st.text_input("Search Announcements")
    announcements = get_all_announcements()
    filtered_anns = {k: v for k, v in announcements.items() 
                    if search_key.lower() in v['title'].lower()}
    
    if not filtered_anns:
        st.info("No announcements available")
    else:
        for ann in reversed(filtered_anns.values()):
            with st.expander(f"{ann['title']} ({ann['timestamp']})"):
                st.write(f"Content: {ann['content']}")
                st.write(f"Author: {ann['author']}")


def show_admin_panel():
    """Display admin panel"""
    st.markdown('<h2 class="main-title">Administrator Center</h2>', unsafe_allow_html=True)
    users = load_json('users.json')
    
    for username in users:
        if username == '1neo9':  # Skip admin account
            continue
        
        user_info = users[username]
        with st.expander(f"User: {username}"):
            st.write(f"Gender: {user_info['gender']}")
            st.write(f"Age: {user_info['age']}")
            st.write(f"Nickname: {user_info['nickname']}")
            
            # Input data
            input_records = get_input_history(username)
            if input_records:
                st.write("User Input Data:")
                st.json(input_records)
            else:
                st.write("No input data available")
            
            # Prediction records
            pred_records = get_prediction_history(username)
            if pred_records:
                last_pred = pred_records[-1]
                st.write(f"Latest Assessment: {last_pred['probability']}%")
                st.write(f"Assessment Time: {last_pred['timestamp']}")
            else:
                st.write("No assessment records")
            
            # Message function
            messages = load_json('messages.json', default={})
            msg = messages.get(username, "")
            new_msg = st.text_input("Message to User", value=msg, key=f"msg_{username}")
            if st.button("Save Message", key=f"save_msg_{username}"):
                messages[username] = new_msg
                if save_json('messages.json', messages):
                    st.success("Message saved successfully")
                else:
                    st.error("Failed to save message")


# ------------------------------
# Navigation menu
# ------------------------------
def render_sidebar_nav():
    """Render sidebar navigation"""
    st.sidebar.markdown("""
    <style>
    .nav-header {
        background-color: #e74c3c;
        color: white;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
        margin-bottom: 15px;
    }
    .nav-btn {
        background-color: #f8f9fa;
        width: 100%;
        text-align: left;
        padding: 10px;
        margin: 5px 0;
        border-radius: 4px;
        border: none;
        cursor: pointer;
    }
    .nav-btn:hover {
        background-color: #e9ecef;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown('<div class="nav-header">Navigation</div>', unsafe_allow_html=True)
    
    # Navigation menu configuration - Personal placed last
    menu = {
        'Data Analysis': '📊 Data Analysis',
        'Health Risk Assessment': '📈 Health Risk Assessment',
        'Announcements & Notices': '📢 Announcements & Notices'
    }
    
    # Admin menu
    if st.session_state.get('is_admin', False):
        menu['Administrator Center'] = '🔐 Administrator Center'
        menu['Announcement Management'] = '📝 Announcement Management'
    
    # Add personal center as last item
    menu['User Profile'] = '👤 User Profile'
    
    # Render navigation buttons
    for page_key, label in menu.items():
        if st.sidebar.button(label, key=page_key, use_container_width=True):
            st.session_state['page'] = page_key
    
    # Default page
    if 'page' not in st.session_state:
        st.session_state['page'] = 'Data Analysis'
    
    return st.session_state['page']


# ------------------------------
# Main function
# ------------------------------
def main():
    # Initialize necessary files and directories
    init_file('users.json', {})
    init_file('announcements.json', {})
    init_file('messages.json', {})
    
    # Create user records directory if it doesn't exist
    if not os.path.exists('user_records'):
        os.makedirs('user_records')
    
    try:
        # Load data and model
        health_data = load_health_data()
        model, X_test, y_test = build_model(health_data)
    except Exception as e:
        st.error(f"Error initializing application: {str(e)}")
        return
    
    # Initialize session state
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
    if 'page' not in st.session_state:
        st.session_state['page'] = 'User Login'
    
    # Set background based on login status
    if st.session_state['logged_in']:
        set_bg_image('background.jpg')
    else:
        set_login_bg('login_bg.png')
    
    # Not logged in - show authentication pages
    if not st.session_state['logged_in']:
        st.markdown('<h1 class="main-title">Health Risk Assessment System</h1>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.session_state['page'] == 'User Login':
                show_login_page()
            elif st.session_state['page'] == 'User Registration':
                show_register_page()
    
    # Logged in - show function pages
    else:
        current_page = render_sidebar_nav()
        
        if current_page == 'Data Analysis':
            show_dashboard(health_data, model, X_test, y_test)
        elif current_page == 'Health Risk Assessment' and not st.session_state['is_admin']:
            show_prediction(model)
        elif current_page == 'User Profile':
            show_user_profile()
        elif current_page == 'Announcement Management' and st.session_state['is_admin']:
            show_announcement_management()
        elif current_page == 'Announcements & Notices':
            show_public_announcements()
        elif current_page == 'Administrator Center' and st.session_state['is_admin']:
            show_admin_panel()


if __name__ == "__main__":
    main()
    
