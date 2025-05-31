import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import base64
import os
import json
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 页面配置
st.set_page_config(page_title="心脏病预测", layout="wide")

# 样式设计
sns.set_style("whitegrid")


def set_background_image(image_path='background.jpg'):
    """
    设置页面背景图为本地图片，并调整其大小和位置
    :param image_path: 图片路径（相对于脚本文件）
    """
    import base64

    with open(image_path, "rb") as f:
        encoded_str = base64.b64encode(f.read()).decode()

    # 添加全局样式，包括背景图片的设置、输入框背景颜色和所有文字的加粗
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/jpg;base64,{encoded_str});
            background-size: contain; /* 调整背景图片大小 */
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-position: 70% center;
        }}

        /* 设置所有输入框的背景颜色为白色 */
        input, textarea {{
            background-color: white !important;
            font-weight: normal !important; /* 保证输入框内的文字不是加粗 */
        }}

        /* 加粗所有其他地方的文字 */
        body, h1, h2, h3, h4, h5, h6, p, span, label, button {{
            font-weight: bold !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


def set_background_image1(image_path):
    """
    将指定图片设置为 Streamlit 页面的全屏背景
    """
    with open(image_path, "rb") as img_file:
        encoded_img = base64.b64encode(img_file.read()).decode()

    background_css = f"""
    <style>
        .stApp {{
            background-image: url(data:image/jpg;base64,{encoded_img});
            background-size: cover;             /* 自动填充整个屏幕 */
            background-position: center;         /* 居中显示 */
            background-repeat: no-repeat;        /* 不重复 */
            background-attachment: fixed;        /* 固定背景，滚动时不移动 */
        }}
    </style>
    """
    st.markdown(background_css, unsafe_allow_html=True)


# 确保 users.json 文件存在
def ensure_users_file_exists():
    if not os.path.exists('users.json'):
        with open('users.json', 'w') as f:
            json.dump({}, f)


# 登录验证函数（检查用户名和密码是否匹配）
def login_user(username, password):
    ensure_users_file_exists()
    with open('users.json', 'r') as f:
        users = json.load(f)
    user = users.get(username)
    if user and user["password"] == password:
        is_admin = username == "1neo9"
        return True, is_admin
    return False, False


def register_user(username, password, gender, age, nickname):
    with open('users.json', 'r') as f:
        users = json.load(f)
    if username in users:
        return False
    users[username] = {
        "password": password,
        "gender": gender,
        "age": age,
        "nickname": nickname,  # 👈 添加 nickname 字段
        "is_admin": False  # 默认普通用户
    }
    with open('users.json', 'w') as f:
        json.dump(users, f)
    return True


# 加载数据
@st.cache_data
def load_and_clean_data():
    df = pd.read_excel('heart_0531.xlsx')
    rows_with_nan = df[df.isnull().any(axis=1)]
    df_cleaned = df.dropna()

    def remove_outliers(df):
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        df_no_outliers = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
        return df_no_outliers

    df_final = remove_outliers(df_cleaned)
    return df_final


# 训练模型
@st.cache_resource
def train_model(df):
    X = df[['age', 'sex', 'trestbps', 'chol', 'fbs', 'thalach', 'exang', 'thal']]
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    param_grid = {
        'n_estimators': [100, 50],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    best_rf = grid_search.best_estimator_

    return best_rf, X_test, y_test


# 增加公告
import os
import json
from datetime import datetime


# 确保公告文件存在
def ensure_announcements_file_exists():
    if not os.path.exists('announcements.json'):
        with open('announcements.json', 'w') as f:
            json.dump({}, f)


# 获取所有公告
def get_announcements():
    ensure_announcements_file_exists()
    try:
        with open('announcements.json', 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}


# 保存公告
def save_announcement(announcement_id, data):
    announcements = get_announcements()
    announcements[announcement_id] = data
    with open('announcements.json', 'w') as f:
        json.dump(announcements, f, indent=4)


# 删除公告
def delete_announcement(announcement_id):
    announcements = get_announcements()
    if announcement_id in announcements:
        del announcements[announcement_id]
        with open('announcements.json', 'w') as f:
            json.dump(announcements, f, indent=4)


# 生成唯一ID（基于当前时间）
def generate_announcement_id():
    return datetime.now().strftime("ANN_%Y%m%d%H%M%S")


def render_admin_announcement():
    st.title("📢 管理员公告管理")

    # 加载已有公告
    announcements = get_announcements()

    # 发布新公告
    st.subheader("📌 发布新公告")
    title = st.text_input("公告标题")
    content = st.text_area("公告内容")
    if st.button("发布"):
        if title.strip() == "" or content.strip() == "":
            st.warning("标题或内容不能为空！")
        else:
            announcement_id = generate_announcement_id()
            data = {
                "title": title,
                "content": content,
                "author": st.session_state.get("current_user"),
                "timestamp": str(datetime.now())
            }
            save_announcement(announcement_id, data)
            st.success("公告已发布！")
            st.rerun()

    # 搜索公告
    search_term = st.text_input("🔍 输入关键词搜索公告标题")
    filtered = {k: v for k, v in announcements.items() if search_term.lower() in v['title'].lower()}

    # 展示公告列表
    st.markdown("---")
    st.subheader("🗂 当前公告列表")
    if not filtered:
        st.info("暂无公告。")
    else:
        for aid, adata in reversed(filtered.items()):
            with st.expander(f"📢 {adata['title']} （{adata['timestamp']}）"):
                st.markdown(f"**内容：**\n{adata['content']}")
                st.markdown(f"*发布人：{adata['author']}*")
                if st.button("🗑 删除", key=f"del_{aid}"):
                    delete_announcement(aid)
                    st.success("公告已删除！")
                    st.rerun()
                if st.button("✏️ 编辑", key=f"edit_{aid}"):
                    st.session_state.editing_announcement = aid
                    st.rerun()

    # 编辑公告功能
    if 'editing_announcement' in st.session_state:
        aid = st.session_state.editing_announcement
        adata = announcements[aid]
        st.markdown("---")
        st.subheader("✍️ 编辑公告")
        new_title = st.text_input("编辑标题", value=adata['title'])
        new_content = st.text_area("编辑内容", value=adata['content'])
        if st.button("更新公告"):
            updated_data = {
                "title": new_title,
                "content": new_content,
                "author": adata['author'],
                "timestamp": adata['timestamp']
            }
            save_announcement(aid, updated_data)
            del st.session_state.editing_announcement
            st.success("公告已更新！")
            st.rerun()
        if st.button("取消编辑"):
            del st.session_state.editing_announcement
            st.rerun()


def render_public_announcement():
    st.title("📢 公告查看区")

    announcements = get_announcements()
    search_term = st.text_input("🔍 输入关键词搜索公告标题")
    filtered = {k: v for k, v in announcements.items() if search_term.lower() in v['title'].lower()}

    if not filtered:
        st.info("暂无公告。")
    else:
        for aid, adata in reversed(filtered.items()):
            with st.expander(f"📢 {adata['title']} （{adata['timestamp']}）"):
                st.markdown(f"**内容：**\n{adata['content']}")
                st.markdown(f"*发布人：{adata['author']}*")


# =============================
# 注册登录
def render_login():
    st.markdown("""
    <div style="background-color:#e0f7ff;padding:20px;border-radius:10px;box-shadow:0 4px 8px rgba(0,0,0,0.2);max-width:400px;margin:auto;text-align:center;">
        <h3>登录</h3>
    """, unsafe_allow_html=True)
    username = st.text_input("用户名")
    password = st.text_input("密码", type="password")
    if st.button("登录"):
        success, is_admin = login_user(username, password)
        if success:
            st.session_state['logged_in'] = True
            st.session_state['current_user'] = username
            st.session_state['is_admin'] = is_admin
            st.session_state['page'] = "数据分析与可视化"  # 跳转到主页
            st.rerun()
        else:
            st.error("用户名或密码错误")
    st.markdown("</div>", unsafe_allow_html=True)


def render_register():
    st.markdown("""
    <div style="background-color:#e0f7ff;padding:20px;border-radius:10px;box-shadow:0 4px 8px rgba(0,0,0,0.2);max-width:400px;margin:auto;text-align:center;">
        <h3>注册</h3>
    """, unsafe_allow_html=True)
    new_username = st.text_input("新用户名")
    new_password = st.text_input("新密码", type="password")
    confirm_password = st.text_input("确认密码", type="password")
    gender = st.selectbox("性别", ["男", "女"])
    age = st.number_input("年龄", min_value=0, max_value=120, value=18)
    nickname = st.text_input("昵称")  # 👈 新增昵称输入

    if st.button("注册"):
        if new_password != confirm_password:
            st.error("两次输入的密码不一致！")
        elif len(new_password) < 6:
            st.warning("密码至少需要6位字符！")
        elif not nickname.strip():  # 检查昵称是否为空
            st.warning("昵称不能为空！")
        else:
            if register_user(new_username, new_password, gender, age, nickname):
                st.success("注册成功，正在跳转登录页面...。")
                # time.sleep(1.5)
                st.session_state['page'] = "登录"
                st.rerun()
            else:
                st.warning("用户名已存在，请换一个。")
    st.markdown("</div>", unsafe_allow_html=True)


def render_login_register():
    st.info("请使用顶部导航栏选择【登录】或【注册】")


# 新增留言管理功能
def ensure_messages_file_exists():
    if not os.path.exists('messages.json'):
        with open('messages.json', 'w') as f:
            json.dump({}, f)


def get_message(username):
    ensure_messages_file_exists()
    with open('messages.json', 'r') as f:
        messages = json.load(f)
    return messages.get(username, "")


def save_message(username, message):
    ensure_messages_file_exists()
    with open('messages.json', 'r') as f:
        messages = json.load(f)
    messages[username] = message
    with open('messages.json', 'w') as f:
        json.dump(messages, f)


# 创建管理员页面
def render_admin_page():
    st.title("🔐 管理员面板")

    with open('users.json', 'r') as f:
        users = json.load(f)

    for username in users:
        if username == "1neo9":  # 跳过管理员自己
            continue

        with st.expander(f"📄 用户：{username}"):
            st.markdown(f"**性别：** {users[username]['gender']}")
            st.markdown(f"**年龄：** {users[username]['age']}")

            # 显示用户输入的数据
            input_file = f"{username}_input_data.json"
            if os.path.exists(input_file):
                with open(input_file, "r") as f:
                    input_data = json.load(f)
                st.markdown("**📌 用户输入数据：**")
                st.json(input_data)
            else:
                st.markdown("**⚠️ 尚未有输入数据记录。**")

            # 显示预测结果
            pred_file = f"{username}_prediction_result.json"
            if os.path.exists(pred_file):
                with open(pred_file, "r") as f:
                    pred_result = json.load(f)
                st.markdown(f"**📈 最近预测概率：** {pred_result['probability']}%")
                st.markdown(f"**🕒 最后预测时间：** {pred_result['timestamp']}")
            else:
                st.markdown("**⚠️ 尚未有预测结果记录。**")

            # 留言功能
            msg = get_message(username)
            new_msg = st.text_input(f"给 {username} 的留言", value=msg, key=f"msg_{username}")
            if st.button(f"保存留言给 {username}", key=f"save_{username}"):
                save_message(username, new_msg)
                st.success("留言已保存！")


def load_user_predictions(username):
    prediction_file = f'{username}_prediction_result.json'
    if os.path.exists(prediction_file):
        with open(prediction_file, 'r') as f:
            return json.load(f)
    return {}


def delete_prediction(username):
    prediction_file = f'{username}_prediction_result.json'
    if os.path.exists(prediction_file):
        os.remove(prediction_file)
        st.success("记录删除成功！")
    else:
        st.warning("没有找到相关记录。")


def get_predictions():
    ensure_predictions_file_exists()  # 确保文件存在
    try:
        with open('predictions.json', 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        # 如果文件损坏或为空，重置为 {}
        print("[警告] predictions.json 文件格式错误，正在重置")
        with open('predictions.json', 'w') as f:
            json.dump({}, f)
        return {}


# 个人资料页面
# =============================
def render_profile():
    st.title("🧾 个人资料")

    username = st.session_state.get("current_user")
    if not username:
        st.warning("请先登录查看个人资料。")
        return

    with open('users.json', 'r') as f:
        users = json.load(f)

    user_info = users.get(username)
    if not user_info:
        st.error("用户信息不存在，请重新登录。")
        return

    st.markdown(f"**用户名：** {username}")
    st.markdown(f"**昵称：** {user_info.get('nickname', username)}")
    st.markdown(f"**性别：** {user_info['gender']}")
    st.markdown(f"**年龄：** {user_info['age']}")
    st.markdown("---")
    st.subheader("✒️ 修改昵称")
    new_nickname = st.text_input("昵称", value=user_info.get("nickname", ""))
    if st.button("保存昵称"):
        users[username]["nickname"] = new_nickname
        with open('users.json', 'w') as f:
            json.dump(users, f)
        st.success("昵称已更新！")
    st.markdown("---")
    st.subheader("🔐 更改密码")

    old_password = st.text_input("当前密码", type="password")
    new_password = st.text_input("新密码", type="password")
    confirm_password = st.text_input("确认新密码", type="password")

    if st.button("更改密码"):
        if old_password != user_info["password"]:
            st.error("当前密码错误！")
        elif new_password != confirm_password:
            st.error("两次输入的新密码不一致！")
        elif new_password.strip() == "":
            st.error("新密码不能为空！")
        else:
            users[username]["password"] = new_password
            with open('users.json', 'w') as f:
                json.dump(users, f)
            st.success("密码修改成功！")
    # 显示管理员留言
    message = get_message(username)
    if message:
        st.markdown("---")
        st.markdown("📢 **管理员留言：**")
        st.info(message)

    # 加载用户的所有预测记录
    def get_all_user_predictions(username):
        prediction_file = f'{username}_prediction_result.json'
        if os.path.exists(prediction_file):
            with open(prediction_file, 'r') as f:
                predictions = json.load(f)
            return predictions  # 返回所有预测记录
        else:
            return []  # 如果没有记录，则返回空列表

    # 在 render_profile 函数内使用
    all_predictions = get_all_user_predictions(username)

    st.markdown("---")
    st.subheader("📋 历史预测记录")

    if not all_predictions:
        st.info("暂无预测记录。")
    else:
        for pred in reversed(all_predictions):  # 倒序显示最新的预测在前
            with st.expander(f"🫀 预测结果：{pred['probability']}% （{pred['timestamp']}）"):
                st.markdown("**📌 输入数据：**")
                try:
                    with open(f"{username}_input_data.json", "r") as f:
                        input_data = json.load(f)
                    st.json(input_data)
                except FileNotFoundError:
                    st.warning("未找到输入数据。")
    st.markdown("---")
    st.subheader("🚪 退出登录")
    if st.button("退出当前账号"):
        # 清除登录状态
        st.session_state['logged_in'] = False
        st.session_state['page'] = "登录与注册"  # 回到登录页
        st.rerun()


import streamlit as st
import streamlit as st


def sidebar_navigation():
    # 自定义CSS样式
    st.markdown("""
    <style>
        .nav-title {
            background-color: #d6eaff; /* 设置为白色或任何你希望的颜色 */
            padding: 10px;
            border: 2px solid #d6eaff; /* 使用边框来框起文字 */
            margin-bottom: 20px;
            text-align: left; /* 左对齐 */
            font-size: 18px;
            font-weight: bold;
            color: #333333;
            width: 100%; /* 确保宽度占满容器 */
            box-sizing: border-box; /* 包括padding和border在内的宽度计算 */
        }
        .nav-button {
            background-color: #d6eaff; /* 浅蓝色 */
            padding: 12px;
            border-radius: 8px;
            margin: 8px 0;
            text-align: left; /* 改成left让图标和文本左对齐 */
            font-weight: bold;
            color: #003366;
            cursor: pointer;
            transition: background-color 0.3s ease;
            width: 100%; /* 确保按钮宽度占满容器 */
            box-sizing: border-box; /* 包括padding和border在内的宽度计算 */
        }
        .nav-button:hover {
            background-color: #a3d0ff;
        }
    </style>
    """, unsafe_allow_html=True)

    # 渲染导航标题
    st.sidebar.markdown('<div class="nav-title">导航</div>', unsafe_allow_html=True)

    # 页面选项和图标
    pages = {
        "数据分析与可视化": "📊 数据分析与可视化",
        "个人信息": "🧾 个人资料",
        "公告发布区": "📢 公告查看区"
    }
    is_admin = st.session_state.get("is_admin", False)  # 安全默认值
    if not is_admin:
        pages["心脏病预测"] = "🫀 心脏病概率预测"
    else:
        # 管理员专属页面
        pages["管理员面板"] = "🔒 管理员面板"
        pages["管理员公告管理"] = "📢 管理员公告管理"
    # 渲染按钮
    for page_key, label in pages.items():
        if st.sidebar.button(label, key=f"nav_{page_key}", use_container_width=True):
            st.session_state["page"] = page_key

    # 默认页面
    if "page" not in st.session_state:
        st.session_state["page"] = "登录与注册"

    return st.session_state.get("page", "登录与注册")


import matplotlib.pyplot as plt
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


def render_visualizations(df, model, X_test, y_test):
    st.title("📊 数据分析与可视化")

    # =====================
    # 字段映射：英文列名 → 简洁中文名（带详细描述）
    # =====================
    var_name_map = {
        'age': '年龄（岁）',
        'sex': '性别（0=女，1=男）',
        'trestbps': '静息血压（mm Hg）',
        'chol': '血清胆固醇（mg/dl）',
        'fbs': '空腹血糖 > 120 mg/dl（0=否，1=是）',
        'thalach': '最大心率',
        'exang': '运动诱发心绞痛（0=否，1=是）',
        'thal': '地中海贫血类型（0=正常，1=固定缺陷，2=可逆缺陷）',
        'target': '是否患病'
    }

    continuous_vars = ['age', 'trestbps', 'chol', 'thalach']
    categorical_vars = ['sex', 'fbs', 'exang', 'thal']  # 只保留指定的分类变量

    # 自动添加中文名称
    def get_chinese_name(var):
        return var_name_map.get(var, var)

    # =====================
    # a) 连续型变量直方图
    # =====================
    st.subheader("a) 连续型变量分布直方图")
    cols = st.columns(2)
    for i, var in enumerate(continuous_vars):
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.histplot(df[var], kde=True, ax=ax, color='skyblue')
        ax.set_title(var)
        cols[i % 2].pyplot(fig)
        cols[i % 2].caption(f"`{var}` → {get_chinese_name(var)}")  # 添加中文注释

    # =====================
    # b) 分类变量饼图（只保留 sex, fbs, exang, thal）
    # =====================
    st.subheader("b) 分类变量分布饼图")
    cols = st.columns(2)
    pie_colors = ['skyblue', 'lightgreen', 'salmon', 'gold', 'violet', 'orange', 'cyan']

    for i, var in enumerate(categorical_vars):
        fig, ax = plt.subplots(figsize=(5, 3))
        value_counts = df[var].value_counts()
        labels = [f"{idx} ({count})" for idx, count in zip(value_counts.index, value_counts)]
        ax.pie(value_counts, labels=labels, autopct='%1.1f%%', colors=pie_colors[:len(value_counts)])
        ax.set_title(var)
        ax.axis('equal')
        cols[i % 2].pyplot(fig)
        cols[i % 2].caption(f"`{var}` → {get_chinese_name(var)}")  # 添加中文注释

    # =====================
    # c) 不同目标下的箱型图
    # =====================
    st.subheader("c) 是否患病 vs 各指标箱型图")
    cols = st.columns(2)
    for i, var in enumerate(continuous_vars):
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.boxplot(x='target', y=var, data=df, ax=ax, palette="Set2")
        ax.set_xlabel('target')
        ax.set_ylabel(var)
        cols[i % 2].pyplot(fig)
        cols[i % 2].caption(f"`{var}` → {get_chinese_name(var)}")  # 添加中文注释

    # =====================
    # d) 相关系数图（仅连续变量）
    # =====================
    st.subheader("d) 连续变量相关系数热力图")
    corr_df = df[continuous_vars].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_df, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=ax,
                xticklabels=continuous_vars,
                yticklabels=continuous_vars)
    st.pyplot(fig)
    # 在相关系数图下方添加匹配关系
    st.markdown("#### 字段说明")
    st.markdown("| 英文字段 | 中文含义 |")
    st.markdown("| --- | --- |")
    for var in continuous_vars:
        st.markdown(f"| `{var}` | {get_chinese_name(var)} |")

    def render_model_performance(X_test, y_test, model):
        st.subheader("e) 模型性能评估")
        # 获取预测结果
        y_pred = model.predict(X_test)

        # 计算各项指标
        report = classification_report(y_test, y_pred, output_dict=True)
        accuracy = report['accuracy']
        recall = report['1']['recall']  # 假设正类为标签1
        precision = report['1']['precision']
        f1_score_val = report['1']['f1-score']

        st.markdown(f"**准确率 Accuracy:** {accuracy:.4f}")
        st.markdown(f"**召回率 Recall (Positive):** {recall:.4f}")
        st.markdown(f"**精确率 Precision (Positive):** {precision:.4f}")
        st.markdown(f"**F1 Score (Positive):** {f1_score_val:.4f}")

        # 绘制混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
        ax.set_xlabel('Prediction Value')
        ax.set_ylabel('Actual Value')
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)

        # 绘制 ROC 曲线
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], 'navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC')
        ax.legend(loc="lower right")
        st.pyplot(fig)

    render_model_performance(X_test, y_test, model)


def ensure_predictions_file_exists():
    if not os.path.exists('predictions.json'):
        with open('predictions.json', 'w') as f:
            json.dump({}, f)


def initialize_files():
    ensure_users_file_exists()
    ensure_messages_file_exists()
    ensure_predictions_file_exists()  # 新增这一行


def render_prediction(model):
    ensure_predictions_file_exists()
    st.title("🫀 心脏病概率预测")
    input_data = {}
    with st.form("prediction_form"):
        st.markdown("请填写以下健康指标以预测是否患有心脏病：")
        col1, col2 = st.columns(2)

        input_fields = {
            "age": "年龄（岁）",
            "sex": "性别（0=女，1=男）",
            "trestbps": "静息血压（mm Hg）",
            "chol": "血清胆固醇（mg/dl）",
            "fbs": "空腹血糖 > 120 mg/dl（0=否，1=是）",
            "thalach": "最大心率",
            "exang": "运动诱发心绞痛（0=否，1=是）",
            "thal": "地中海贫血类型（0=正常，1=固定缺陷，2=可逆缺陷）"
        }

        for i, (key, label) in enumerate(input_fields.items()):
            with col1 if i % 2 == 0 else col2:
                input_data[key] = st.number_input(label=label, value=0, step=1)

        submit_button = st.form_submit_button("预测")

    if submit_button:
        input_df = pd.DataFrame([input_data])
        proba = model.predict_proba(input_df)[0][1]
        percent = f"{proba * 100:.2f}"
        st.success(f"预测患心脏病的概率为：**{percent}%**")

        username = st.session_state.get("current_user")
        if username:
            timestamp = str(pd.Timestamp.now())

            # 构建预测记录
            prediction_result = {
                "probability": percent,
                "timestamp": timestamp
            }

            # ========== 追加写入历史记录 ==========
            prediction_file = f'{username}_prediction_result.json'

            # 如果文件存在，先读取已有数据，再添加新记录
            predictions = []
            if os.path.exists(prediction_file):
                with open(prediction_file, 'r') as f:
                    predictions = json.load(f)

            predictions.append(prediction_result)

            with open(prediction_file, 'w') as f:
                json.dump(predictions, f)

            # ========== 同时保存本次输入的数据 ==========
            input_file = f'{username}_input_data.json'
            input_history = []
            if os.path.exists(input_file):
                with open(input_file, 'r') as f:
                    input_history = json.load(f)

            # 将当前输入数据加上时间戳存储（可选）
            input_with_time = input_data.copy()
            input_with_time['_timestamp'] = timestamp
            input_history.append(input_with_time)

            with open(input_file, 'w') as f:
                json.dump(input_history, f)


def main():
    # 设置根据登录状态动态更换背景图
    if st.session_state.get('logged_in', False):
        set_background_image('background.jpg')  # 已登录后的页面用这个
    else:
        set_background_image1('login_bg.jpg')  # 登录/注册页面用这个
    initialize_files()
    df = load_and_clean_data()
    model, X_test, y_test = train_model(df)
    # 初始化 session_state 状态
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False

    if 'page' not in st.session_state:
        st.session_state['page'] = "登录与注册"  # 默认进入登录页

    # 如果未登录，只允许访问登录/注册页
    if not st.session_state.get('logged_in', False):
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("<h2 style='text-align:center;'>登录与注册</h2>", unsafe_allow_html=True)
            if st.session_state['page'] == "登录":
                render_login()
            elif st.session_state['page'] == "注册":
                render_register()
    # 已登录，进入主功能页面
    else:
        page = sidebar_navigation()

        if page == "数据分析与可视化":
            render_visualizations(df, model, X_test, y_test)
        elif page == "心脏病预测":
            render_prediction(model)
        elif page == "个人信息":
            render_profile()
        elif page == "管理员公告管理":
            if st.session_state.get("is_admin", False):
                render_admin_announcement()
            else:
                st.warning("您没有权限访问此页面。")
        elif page == "公告发布区":
            render_public_announcement()
        elif page == "管理员面板":
            if st.session_state.get("is_admin", False):
                render_admin_page()
            else:
                st.warning("您没有权限访问此页面。")


if __name__ == "__main__":
    main()
