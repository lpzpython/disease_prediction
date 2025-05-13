#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import json

# 页面配置
st.set_page_config(page_title="心脏病的预测", layout="wide")

# 样式设计
sns.set_style("whitegrid")

# 加载数据
@st.cache_data
def load_data():
    df = pd.read_excel('heart_0513.xlsx')
    return df

# 加载模型
def load_model():
    return joblib.load('random_forest_model.joblib')

# 用户登录管理
def login_user(username, password):
    if not os.path.exists('users.json'):
        with open('users.json', 'w') as f:
            json.dump({}, f)
    with open('users.json', 'r') as f:
        users = json.load(f)
    return users.get(username) == password

def register_user(username, password):
    with open('users.json', 'r') as f:
        users = json.load(f)
    if username in users:
        return False
    users[username] = password
    with open('users.json', 'w') as f:
        json.dump(users, f)
    return True

# 侧边栏导航
def sidebar_navigation():
    st.sidebar.title("导航")
    selection = st.sidebar.radio("选择页面", ["登录", "注册", "数据分析与可视化", "心脏病预测"])
    return selection

# 页面部分
def render_login():
    st.subheader("登录")
    username = st.text_input("用户名")
    password = st.text_input("密码", type='password')
    if st.button("登录"):
        if login_user(username, password):
            st.session_state['logged_in'] = True
            st.success("登录成功")
        else:
            st.error("用户名或密码错误")

def render_register():
    st.subheader("注册")
    new_username = st.text_input("新用户名")
    new_password = st.text_input("新密码", type='password')
    if st.button("注册"):
        if register_user(new_username, new_password):
            st.success("注册成功，请去登录")
        else:
            st.warning("用户名已存在")

def render_visualizations(df):
    st.title("📊 数据分析与可视化")

    continuous_vars = ['age', 'trestbps', 'chol', 'thalach']
    categorical_vars = [col for col in df.columns if col not in continuous_vars and col != 'target']

    st.subheader("a) 连续型变量的直方图")
    cols = st.columns(2)
    for i, var in enumerate(continuous_vars):
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.histplot(df[var], kde=True, ax=ax, color='skyblue')
        cols[i % 2].pyplot(fig)

    st.subheader("b) 分类变量分布饼图")
    cols = st.columns(2)
    for i, var in enumerate(categorical_vars):
        fig, ax = plt.subplots(figsize=(5, 3))
        df[var].value_counts().plot.pie(autopct='%1.1f%%', ax=ax, colors=sns.color_palette("pastel"))
        ax.set_ylabel('')
        cols[i % 2].pyplot(fig)

    st.subheader("c) 不同目标下的箱型图")
    cols = st.columns(2)
    for i, var in enumerate(continuous_vars):
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.boxplot(x='target', y=var, data=df, ax=ax, palette="Set2")
        cols[i % 2].pyplot(fig)

    st.subheader("d) 变量相关系数图")
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=ax)
    st.pyplot(fig)

def render_prediction(df, model):
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
            "restecg": "静息心电图结果（0~2）",
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
        st.success(f"预测患心脏病的概率为：**{proba * 100:.2f}%**")

# 主函数逻辑
def main():
    df = load_data()
    model = load_model()

    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False

    page = sidebar_navigation()

    if page == "登录":
        render_login()
    elif page == "注册":
        render_register()
    elif st.session_state['logged_in']:
        if page == "数据分析与可视化":
            render_visualizations(df)
        elif page == "心脏病预测":
            render_prediction(df, model)
    else:
        st.warning("请先登录以访问此页面。")

if __name__ == "__main__":
    main()

