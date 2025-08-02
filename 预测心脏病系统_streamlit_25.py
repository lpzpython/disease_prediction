import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import base64
import os
import json
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# 页面基础配置
st.set_page_config(page_title="心脏健康评估系统", layout="wide")
sns.set_style("ticks")  # 调整默认样式


# ------------------------------
# 样式与背景设置
# ------------------------------
def set_bg_image(img_path):
    """设置页面背景图片"""
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
            border-left: 4px solid #3498db;
            padding-left: 10px;
            margin-top: 1.5rem;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


def set_login_bg(img_path):
    """设置登录页面背景"""
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
# 数据文件管理
# ------------------------------
def init_file(file_path, default_content):
    """初始化文件（若不存在则创建）"""
    if not os.path.exists(file_path):
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(default_content, f, ensure_ascii=False)


def load_json(file_path):
    """读取JSON文件内容"""
    init_file(file_path, {})
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}


def save_json(file_path, data):
    """保存数据到JSON文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


# ------------------------------
# 用户认证相关
# ------------------------------
def verify_user(username, password):
    """验证用户登录信息"""
    users = load_json('users.json')
    user_info = users.get(username)
    if user_info and user_info['password'] == password:
        return True, username == '1neo9'  # (登录成功, 是否管理员)
    return False, False


def add_new_user(username, password, gender, age, nickname):
    """注册新用户"""
    users = load_json('users.json')
    if username in users:
        return False  # 用户名已存在
    users[username] = {
        'password': password,
        'gender': gender,
        'age': age,
        'nickname': nickname,
        'is_admin': False
    }
    save_json('users.json', users)
    return True


# ------------------------------
# 数据处理与模型训练
# ------------------------------
@st.cache_data
def load_health_data():
    """加载并清洗心脏健康数据"""
    # 加载数据并处理缺失值
    raw_data = pd.read_excel('heart_0531.xlsx')
    clean_data = raw_data.dropna()
    
    # 移除异常值
    def drop_outliers(df):
        q1 = df.quantile(0.25)
        q3 = df.quantile(0.75)
        iqr = q3 - q1
        return df[~((df < (q1 - 1.5 * iqr)) | (df > (q3 + 1.5 * iqr))).any(axis=1)]
    
    return drop_outliers(clean_data)


@st.cache_resource
def build_model(health_data):
    """训练随机森林分类模型"""
    # 特征与目标变量
    features = ['age', 'sex', 'trestbps', 'chol', 'fbs', 'thalach', 'exang', 'thal']
    X = health_data[features]
    y = health_data['target']
    
    # 拆分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42
    )
    
    # 网格搜索优化模型
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
# 公告管理功能
# ------------------------------
def get_all_announcements():
    """获取所有公告"""
    return load_json('announcements.json')


def add_announcement(title, content, author):
    """添加新公告"""
    announcements = get_all_announcements()
    ann_id = f"ANN_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    announcements[ann_id] = {
        'title': title,
        'content': content,
        'author': author,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    save_json('announcements.json', announcements)
    return ann_id


def update_announcement(ann_id, new_title, new_content):
    """更新公告内容"""
    announcements = get_all_announcements()
    if ann_id in announcements:
        announcements[ann_id]['title'] = new_title
        announcements[ann_id]['content'] = new_content
        save_json('announcements.json', announcements)
        return True
    return False


def remove_announcement(ann_id):
    """删除公告"""
    announcements = get_all_announcements()
    if ann_id in announcements:
        del announcements[ann_id]
        save_json('announcements.json', announcements)
        return True
    return False


# ------------------------------
# 页面渲染 - 认证相关
# ------------------------------
def show_login_page():
    """显示登录页面"""
    with st.container():
        st.markdown('<div class="login-card">', unsafe_allow_html=True)
        st.subheader("用户登录")
        
        username = st.text_input("请输入用户名")
        password = st.text_input("请输入密码", type="password")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("登录", use_container_width=True):
                success, is_admin = verify_user(username, password)
                if success:
                    st.session_state.update({
                        'logged_in': True,
                        'current_user': username,
                        'is_admin': is_admin,
                        'page': '数据概览与分析'
                    })
                    st.rerun()
                else:
                    st.error("用户名或密码不正确")
        
        with col2:
            if st.button("注册新账号", use_container_width=True):
                st.session_state['page'] = '用户注册'
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


def show_register_page():
    """显示注册页面"""
    with st.container():
        st.markdown('<div class="login-card">', unsafe_allow_html=True)
        st.subheader("用户注册")
        
        new_user = st.text_input("设置用户名")
        new_pwd = st.text_input("设置密码（至少6位）", type="password")
        confirm_pwd = st.text_input("确认密码", type="password")
        gender = st.selectbox("性别", ["男", "女"])
        age = st.number_input("年龄", min_value=0, max_value=120, value=18)
        nickname = st.text_input("设置昵称（必填）")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("完成注册", use_container_width=True):
                if new_pwd != confirm_pwd:
                    st.error("两次输入的密码不一致")
                elif len(new_pwd) < 6:
                    st.warning("密码长度不能少于6位")
                elif not nickname.strip():
                    st.warning("昵称不能为空")
                else:
                    if add_new_user(new_user, new_pwd, gender, age, nickname):
                        st.success("注册成功，即将跳转到登录页")
                        st.session_state['page'] = '用户登录'
                        st.rerun()
                    else:
                        st.warning("用户名已存在，请更换")
        
        with col2:
            if st.button("返回登录", use_container_width=True):
                st.session_state['page'] = '用户登录'
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


# ------------------------------
# 页面渲染 - 功能页面
# ------------------------------
def show_dashboard(health_data, model, X_test, y_test):
    """显示数据分析仪表盘"""
    st.markdown('<h2 class="main-title">心脏健康数据分析</h2>', unsafe_allow_html=True)
    
    # 数据字段说明
    field_explain = {
        'age': '年龄（岁）',
        'sex': '性别（0=女，1=男）',
        'trestbps': '静息血压（mm Hg）',
        'chol': '血清胆固醇（mg/dl）',
        'fbs': '空腹血糖 > 120 mg/dl（0=否，1=是）',
        'thalach': '最大心率',
        'exang': '运动诱发心绞痛（0=否，1=是）',
        'thal': '地中海贫血类型（0=正常，1=固定缺陷，2=可逆缺陷）',
        'target': '是否患病（1=是，0=否）'
    }
    
    # 连续变量分布
    st.markdown('<h3 class="section-title">1. 连续指标分布</h3>', unsafe_allow_html=True)
    cont_vars = ['age', 'trestbps', 'chol', 'thalach']
    cols = st.columns(2)
    for i, var in enumerate(cont_vars):
        with cols[i % 2]:
            fig, ax = plt.subplots(figsize=(5, 3))
            sns.histplot(health_data[var], kde=True, ax=ax, color='#3498db')
            ax.set_title(f'{var} - {field_explain[var]}')
            st.pyplot(fig)
    
    # 分类变量分布
    st.markdown('<h3 class="section-title">2. 分类指标分布</h3>', unsafe_allow_html=True)
    cat_vars = ['sex', 'fbs', 'exang', 'thal']
    cols = st.columns(2)
    for i, var in enumerate(cat_vars):
        with cols[i % 2]:
            fig, ax = plt.subplots(figsize=(5, 3))
            counts = health_data[var].value_counts()
            ax.pie(counts, labels=counts.index, autopct='%1.1f%%', 
                   colors=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'])
            ax.set_title(f'{var} - {field_explain[var]}')
            st.pyplot(fig)
    
    # 患病情况与指标关系
    st.markdown('<h3 class="section-title">3. 患病情况与指标关系</h3>', unsafe_allow_html=True)
    cols = st.columns(2)
    for i, var in enumerate(cont_vars):
        with cols[i % 2]:
            fig, ax = plt.subplots(figsize=(5, 3))
            sns.boxplot(x='target', y=var, data=health_data, ax=ax, palette=['#2ecc71', '#e74c3c'])
            ax.set_xlabel('是否患病')
            ax.set_ylabel(f'{var} - {field_explain[var]}')
            st.pyplot(fig)
    
    # 相关性分析
    st.markdown('<h3 class="section-title">4. 指标相关性分析</h3>', unsafe_allow_html=True)
    corr_data = health_data[cont_vars].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_data, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
    st.pyplot(fig)
    
    # 模型性能
    st.markdown('<h3 class="section-title">5. 模型性能评估</h3>', unsafe_allow_html=True)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # 性能指标
    report = classification_report(y_test, y_pred, output_dict=True)
    st.write(f"准确率: {report['accuracy']:.4f}")
    st.write(f"患病识别率: {report['1']['recall']:.4f}")
    st.write(f"预测精准度: {report['1']['precision']:.4f}")
    
    # 混淆矩阵
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('预测结果')
    ax.set_ylabel('实际结果')
    st.pyplot(fig)
    
    # ROC曲线
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fpr, tpr, color='#e74c3c', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlabel('假阳性率')
    ax.set_ylabel('真阳性率')
    ax.legend()
    st.pyplot(fig)


def show_prediction(model):
    """显示心脏病预测页面"""
    st.markdown('<h2 class="main-title">心脏健康风险评估</h2>', unsafe_allow_html=True)
    
    with st.form("pred_form"):
        st.write("请输入以下健康指标，系统将评估心脏病风险：")
        col1, col2 = st.columns(2)
        
        # 输入字段
        input_vals = {}
        fields = [
            ('age', '年龄（岁）', 50),
            ('sex', '性别（0=女，1=男）', 0),
            ('trestbps', '静息血压（mm Hg）', 120),
            ('chol', '血清胆固醇（mg/dl）', 200),
            ('fbs', '空腹血糖>120mg/dl（0=否，1=是）', 0),
            ('thalach', '最大心率', 150),
            ('exang', '运动诱发心绞痛（0=否，1=是）', 0),
            ('thal', '地中海贫血类型（0=正常，1=固定，2=可逆）', 0)
        ]
        
        for i, (key, label, default) in enumerate(fields):
            with col1 if i % 2 == 0 else col2:
                input_vals[key] = st.number_input(label, value=default, step=1)
        
        submit_btn = st.form_submit_button("开始评估", use_container_width=True)
    
    if submit_btn:
        # 生成预测结果
        input_df = pd.DataFrame([input_vals])
        risk_prob = model.predict_proba(input_df)[0][1] * 100
        st.success(f"心脏病风险评估结果：**{risk_prob:.2f}%**")
        
        # 保存记录
        username = st.session_state['current_user']
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # 保存预测结果
        pred_file = f'{username}_risk_records.json'
        pred_records = load_json(pred_file)
        pred_records.append({
            'probability': f'{risk_prob:.2f}',
            'timestamp': timestamp
        })
        save_json(pred_file, pred_records)
        
        # 保存输入数据
        input_file = f'{username}_input_data.json'
        input_records = load_json(input_file)
        input_data = input_vals.copy()
        input_data['timestamp'] = timestamp
        input_records.append(input_data)
        save_json(input_file, input_records)


def show_user_profile():
    """显示用户个人资料页面"""
    st.markdown('<h2 class="main-title">个人中心</h2>', unsafe_allow_html=True)
    username = st.session_state['current_user']
    users = load_json('users.json')
    user_info = users.get(username)
    
    if not user_info:
        st.error("未找到用户信息，请重新登录")
        return
    
    # 基本信息
    st.markdown('<h3 class="section-title">基本信息</h3>', unsafe_allow_html=True)
    st.write(f"用户名: {username}")
    st.write(f"昵称: {user_info['nickname']}")
    st.write(f"性别: {user_info['gender']}")
    st.write(f"年龄: {user_info['age']}")
    
    # 修改昵称
    st.markdown('<h3 class="section-title">修改昵称</h3>', unsafe_allow_html=True)
    new_nick = st.text_input("新昵称", value=user_info['nickname'])
    if st.button("保存昵称"):
        users[username]['nickname'] = new_nick
        save_json('users.json', users)
        st.success("昵称已更新")
    
    # 修改密码
    st.markdown('<h3 class="section-title">修改密码</h3>', unsafe_allow_html=True)
    old_pwd = st.text_input("当前密码", type="password")
    new_pwd = st.text_input("新密码", type="password")
    confirm_pwd = st.text_input("确认新密码", type="password")
    
    if st.button("更新密码"):
        if old_pwd != user_info['password']:
            st.error("当前密码不正确")
        elif new_pwd != confirm_pwd:
            st.error("两次输入的新密码不一致")
        elif not new_pwd:
            st.error("新密码不能为空")
        else:
            users[username]['password'] = new_pwd
            save_json('users.json', users)
            st.success("密码已更新")
    
    # 管理员留言
    msg = load_json('messages.json').get(username, "")
    if msg:
        st.markdown('<h3 class="section-title">管理员留言</h3>', unsafe_allow_html=True)
        st.info(msg)
    
    # 历史记录
    st.markdown('<h3 class="section-title">评估历史</h3>', unsafe_allow_html=True)
    pred_file = f'{username}_risk_records.json'
    pred_records = load_json(pred_file)
    
    if not pred_records:
        st.info("暂无评估记录")
    else:
        for rec in reversed(pred_records):
            with st.expander(f"评估时间: {rec['timestamp']} | 风险值: {rec['probability']}%"):
                try:
                    input_data = load_json(f'{username}_input_data.json')
                    # 找到对应时间的输入数据
                    for data in reversed(input_data):
                        if data['timestamp'] == rec['timestamp']:
                            st.json({k: v for k, v in data.items() if k != 'timestamp'})
                            break
                except:
                    st.warning("未找到对应输入数据")
    
    # 退出登录
    st.markdown('<h3 class="section-title">账户操作</h3>', unsafe_allow_html=True)
    if st.button("退出登录"):
        st.session_state['logged_in'] = False
        st.session_state['page'] = '用户登录'
        st.rerun()


def show_announcement_management():
    """显示管理员公告管理页面"""
    st.markdown('<h2 class="main-title">公告管理</h2>', unsafe_allow_html=True)
    
    # 发布新公告
    st.markdown('<h3 class="section-title">发布新公告</h3>', unsafe_allow_html=True)
    with st.form("new_ann_form"):
        title = st.text_input("公告标题")
        content = st.text_area("公告内容")
        if st.form_submit_button("发布公告"):
            if not title or not content:
                st.warning("标题和内容不能为空")
            else:
                add_announcement(title, content, st.session_state['current_user'])
                st.success("公告发布成功")
                st.rerun()
    
    # 搜索与筛选
    search_key = st.text_input("搜索公告标题")
    announcements = get_all_announcements()
    filtered_anns = {k: v for k, v in announcements.items() 
                    if search_key.lower() in v['title'].lower()}
    
    # 公告列表
    st.markdown('<h3 class="section-title">公告列表</h3>', unsafe_allow_html=True)
    if not filtered_anns:
        st.info("暂无公告")
    else:
        for ann_id, ann in reversed(filtered_anns.items()):
            with st.expander(f"{ann['title']} ({ann['timestamp']})"):
                st.write(f"内容: {ann['content']}")
                st.write(f"发布人: {ann['author']}")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("删除", key=f"del_{ann_id}"):
                        remove_announcement(ann_id)
                        st.success("公告已删除")
                        st.rerun()
                with col2:
                    if st.button("编辑", key=f"edit_{ann_id}"):
                        st.session_state['edit_ann_id'] = ann_id
                        st.rerun()
    
    # 编辑公告
    if 'edit_ann_id' in st.session_state:
        ann_id = st.session_state['edit_ann_id']
        ann = announcements.get(ann_id)
        if ann:
            st.markdown('<h3 class="section-title">编辑公告</h3>', unsafe_allow_html=True)
            new_title = st.text_input("公告标题", value=ann['title'])
            new_content = st.text_area("公告内容", value=ann['content'])
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("保存修改"):
                    update_announcement(ann_id, new_title, new_content)
                    del st.session_state['edit_ann_id']
                    st.success("公告已更新")
                    st.rerun()
            with col2:
                if st.button("取消编辑"):
                    del st.session_state['edit_ann_id']
                    st.rerun()


def show_public_announcements():
    """显示公共公告页面"""
    st.markdown('<h2 class="main-title">公告通知</h2>', unsafe_allow_html=True)
    
    search_key = st.text_input("搜索公告")
    announcements = get_all_announcements()
    filtered_anns = {k: v for k, v in announcements.items() 
                    if search_key.lower() in v['title'].lower()}
    
    if not filtered_anns:
        st.info("暂无公告")
    else:
        for ann in reversed(filtered_anns.values()):
            with st.expander(f"{ann['title']} ({ann['timestamp']})"):
                st.write(f"内容: {ann['content']}")
                st.write(f"发布人: {ann['author']}")


def show_admin_panel():
    """显示管理员面板"""
    st.markdown('<h2 class="main-title">管理员中心</h2>', unsafe_allow_html=True)
    users = load_json('users.json')
    
    for username in users:
        if username == '1neo9':  # 跳过管理员自身
            continue
        
        user_info = users[username]
        with st.expander(f"用户: {username}"):
            st.write(f"性别: {user_info['gender']}")
            st.write(f"年龄: {user_info['age']}")
            st.write(f"昵称: {user_info['nickname']}")
            
            # 输入数据
            input_file = f'{username}_input_data.json'
            if os.path.exists(input_file):
                st.write("用户输入数据:")
                st.json(load_json(input_file))
            else:
                st.write("暂无输入数据")
            
            # 预测记录
            pred_file = f'{username}_risk_records.json'
            if os.path.exists(pred_file):
                pred_records = load_json(pred_file)
                if pred_records:
                    last_pred = pred_records[-1]
                    st.write(f"最近评估: {last_pred['probability']}%")
                    st.write(f"评估时间: {last_pred['timestamp']}")
                else:
                    st.write("暂无评估记录")
            else:
                st.write("暂无评估记录")
            
            # 留言功能
            msg = load_json('messages.json').get(username, "")
            new_msg = st.text_input("给用户的留言", value=msg, key=f"msg_{username}")
            if st.button("保存留言", key=f"save_msg_{username}"):
                messages = load_json('messages.json')
                messages[username] = new_msg
                save_json('messages.json', messages)
                st.success("留言已保存")


# ------------------------------
# 导航菜单
# ------------------------------
def render_sidebar_nav():
    """渲染侧边栏导航"""
    st.sidebar.markdown("""
    <style>
    .nav-header {
        background-color: #3498db;
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
    
    st.sidebar.markdown('<div class="nav-header">功能导航</div>', unsafe_allow_html=True)
    
    # 导航菜单配置
    menu = {
        '数据概览与分析': '📊 数据概览与分析',
        '个人中心': '👤 个人中心',
        '公告通知': '📢 公告通知'
    }
    
    # 管理员菜单
    if st.session_state.get('is_admin', False):
        menu['管理员中心'] = '🔐 管理员中心'
        menu['公告管理'] = '📝 公告管理'
    else:
        menu['健康风险评估'] = '❤️ 健康风险评估'
    
    # 渲染导航按钮
    for page_key, label in menu.items():
        if st.sidebar.button(label, key=page_key, use_container_width=True):
            st.session_state['page'] = page_key
    
    # 默认页面
    if 'page' not in st.session_state:
        st.session_state['page'] = '数据概览与分析'
    
    return st.session_state['page']


# ------------------------------
# 主函数
# ------------------------------
def main():
    # 初始化必要文件
    init_file('users.json', {})
    init_file('announcements.json', {})
    init_file('messages.json', {})
    
    # 加载数据与模型
    health_data = load_health_data()
    model, X_test, y_test = build_model(health_data)
    
    # 初始化会话状态
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
    if 'page' not in st.session_state:
        st.session_state['page'] = '用户登录'
    
    # 根据登录状态设置背景
    if st.session_state['logged_in']:
        set_bg_image('background.jpg')
    else:
        set_login_bg('login_bg.png')
    
    # 未登录状态 - 显示认证页面
    if not st.session_state['logged_in']:
        st.markdown('<h1 class="main-title">心脏健康评估系统</h1>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.session_state['page'] == '用户登录':
                show_login_page()
            elif st.session_state['page'] == '用户注册':
                show_register_page()
    
    # 已登录状态 - 显示功能页面
    else:
        current_page = render_sidebar_nav()
        
        if current_page == '数据概览与分析':
            show_dashboard(health_data, model, X_test, y_test)
        elif current_page == '健康风险评估' and not st.session_state['is_admin']:
            show_prediction(model)
        elif current_page == '个人中心':
            show_user_profile()
        elif current_page == '公告管理' and st.session_state['is_admin']:
            show_announcement_management()
        elif current_page == '公告通知':
            show_public_announcements()
        elif current_page == '管理员中心' and st.session_state['is_admin']:
            show_admin_panel()


if __name__ == "__main__":
    main()
