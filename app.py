import streamlit as st
import pandas as pd

# global variable
feedback_dataframe = pd.DataFrame(columns=['Name', 'Country', 'Mail',  'Course Suggestion', 'Feedback', 'Resume'])
admission_confidence_dataframe = pd.DataFrame(columns=['Name', 'Confidence'])
courses_confidence_dataframe = pd.DataFrame(columns=['Name', 'Confidence'])

def intro():
    import streamlit as st

    st.write("# Welcome to UMBC Course Recommender! 👋")
    st.sidebar.success("Select a tool above.")

    st.markdown(
        """
        UMBC Course recommender helps you find the right course aligned with your interests and goals
        & helps you learn if we are a good fit for you!
        **👈 Select a demo from the sidebar** to explore for yourself!
    """
    )

def admission_selection():
    import streamlit as st
    import pandas as pd
    from joblib import load
    import numpy as np
    from scipy.stats import skew
    import time
    import os

    st.markdown(f"# {list(page_names_to_funcs.keys())[2]}")
    st.write(
        """
        This tool shows your chances of getting accepted into UMBC provided your standard test scores 🎓😊.
        """
    )

    # Utils functions
    def make_prediction(model, scaler, program_name, department, gpa, gre_score, toefl_score):
        # Create a DataFrame from the inputs
        data = pd.DataFrame({
            'Program Name': [program_name],
            'Department': [department],
            'GPA': [gpa],
            'GRE Score': [gre_score],
            'TOEFL Score': [toefl_score]
        })
        
        # Apply the same preprocessing as during training
        categorical_features = ['Program Name', 'Department']
        numerical_features = ['GPA', 'GRE Score', 'TOEFL Score']

        for col in numerical_features:
            if skew(data[col]) > 0 or skew(data[col]) < 0:
                data[col] = np.log1p(data[col])
        
        # Transform categorical data with OneHotEncoder
        data_encoded = pd.get_dummies(data, columns=categorical_features)
        training_columns = ['GPA', 'GRE Score', 'TOEFL Score',
        'Program Name_Chemical & Biochemical Engineering',
        'Program Name_Computer Engineering', 'Program Name_Computer Science',
        'Program Name_Cybersecurity', 'Program Name_Data Science',
        'Program Name_Electrical Engineering',
        'Program Name_Engineering Management',
        'Program Name_Environmental Engineering',
        'Program Name_Health Information Technology',
        'Program Name_Human-Centered Computing',
        'Department_Chemical & Biochemical Engineering',
        'Department_Civil & Environmental Engineering',
        'Department_Computer Science',
        'Department_Computer Science & Electrical Engineering',
        'Department_Electrical Engineering',
        'Department_Engineering Management', 'Department_Information Systems']
        
        data_encoded = data_encoded.reindex(columns=training_columns, fill_value=0)
        X_test_scaled = scaler.transform(data_encoded)

        prediction = model.predict(X_test_scaled)
        confidences = model.predict_proba(X_test_scaled)
        
        return prediction, confidences[0]

    def load_artifacts(model_path, scaler_path):
        model = load(model_path)
        scaler = load(scaler_path)
        return model, scaler
    
    @st.cache_data
    def load_data(grades_path):
        # admission
        grades_dataset = pd.read_csv(grades_path)
        program_names = grades_dataset['Program Name'].unique().tolist()
        department_names = grades_dataset['Department'].unique().tolist()

        return program_names, department_names
    
    grades_path = r"/Users/vinayvarma/Desktop/streamlit-app/data/admissions_acceptance_dataset.csv"
    model_path = r"/Users/vinayvarma/Desktop/streamlit-app/models/admission_ensemble.joblib" 
    scaler_path = r"/Users/vinayvarma/Desktop/streamlit-app/models/scaler.joblib"

    model, scaler = load_artifacts(model_path, scaler_path)

    program_names, department_names = load_data(grades_path)
    
    # Example inputs
    name = st.text_input('Your Name')
    program_name = st.selectbox("Select your Program", options=program_names)
    department = st.selectbox("Select your Department", options=department_names)
    gpa = st.number_input("Enter your GPA",min_value=0.0, max_value=4.0, value=3.5)
    gre_score = st.number_input("Enter your GRE score", max_value=340, value=315)
    toefl_score = st.number_input("Enter your TOEFL score", max_value=120, value=107)
    admission_result = 0

    # Predict using the function
    if st.button('Admission Prediction'):
        with st.spinner('Prediction result'):
            time.sleep(3)
            pred, conf = make_prediction(model, scaler, program_name, department, gpa, gre_score, toefl_score)
            rejection_reasons = []
            if gre_score < 300 or gpa < 2.5 or toefl_score < 80:
                # st.markdown(f"<div style='background-color:#F44336; color:white; padding:10px; border-radius:8px; font-weight:bold;'>Not Admitted</div>", unsafe_allow_html=True)
                if gre_score < 300:
                    rejection_reasons.append("GRE score below the minimum threshold of 300.")
                if gpa < 2.5:
                    rejection_reasons.append("GPA below the minimum threshold of 2.5.")
                if toefl_score < 80:
                    rejection_reasons.append("TOEFL score below the minimum threshold of 80.")
                if not rejection_reasons:
                    rejection_reasons.append("Meets all individual score thresholds but does not fit the overall profile.")
                rejection_html = "<ul>" + "".join(f"<li>{reason}</li>" for reason in rejection_reasons) + "</ul>"
                st.markdown(f"<div style='background-color:#F44336; color:white; padding:10px; border-radius:8px; font-weight:bold;'>Not Admitted: {rejection_html}</div>", unsafe_allow_html=True)
            else:
                if pred[0] == 1:
                    st.markdown(f"<div style='background-color:#4CAF50; color:white; padding:10px; border-radius:8px; font-weight:bold;'>Admitted</div>", unsafe_allow_html=True)
                    admission_result = 1
                else:
                    rejection_reasons.append("Meets all individual score thresholds but does not fit the overall profile.")
                    rejection_html = "<ul>" + "".join(f"<li>{reason}</li>" for reason in rejection_reasons) + "</ul>"
                    st.markdown(f"<div style='background-color:#F44336; color:white; padding:10px; border-radius:8px; font-weight:bold;'>Not Admitted: {rejection_html}</div>", unsafe_allow_html=True)

            confidence_row = pd.DataFrame([[name, conf[1], admission_result]], columns=['Name', 'Confidence', 'Admitted'])
            header = not os.path.exists('admission_confidence.csv')
            confidence_row.to_csv('admission_confidence.csv', mode='a', header=header, index=False)
                


def course_recommendation():
    # Imports
    import streamlit as st
    import pandas as pd
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    from sentence_transformers import SentenceTransformer
    import time
    import os

    st.markdown(f'# {list(page_names_to_funcs.keys())[1]}')
    st.write(
        """
        This tool will help you select and compare courses based on your interests and career goals. Enjoy!
        """
    )

    # Utils functions
    def recommend_courses(interests, career_goals, course_df):
        # Combine interests and career goals into a single profile text
        profile_text = interests + " " + career_goals
        
        # Load a pre-trained sentence transformer model (BERT-based)
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Embed the profile text and all course descriptions
        profile_vector = model.encode([profile_text])
        course_vectors = model.encode(course_df['Matched Courses'].tolist())
        
        # Calculate cosine similarity between profile vector and course vectors
        similarity_scores = cosine_similarity(profile_vector, course_vectors)
        
        # Get indices of courses sorted by similarity (highest first)
        top_indices = similarity_scores.argsort()[0][::-1]
        
        # Collect the top two unique recommendations based on a similarity threshold
        unique_courses = []
        seen_descriptions = []
        similarity_values = []
        for index in top_indices:
            if len(unique_courses) == 2:
                break
            course_description = course_df['Matched Courses'].iloc[index]
            current_similarity = similarity_scores[0][index]
            if not any(cosine_similarity([model.encode(course_description)], [model.encode(seen)]) > 0.9 for seen in seen_descriptions):
                unique_courses.append(course_description)
                seen_descriptions.append(course_description)
                similarity_values.append(current_similarity)
        
        return unique_courses, similarity_values
    
    @st.cache_data
    def load_data(courses_path):
        # Courses
        courses_dataset = pd.read_csv(courses_path)
        unique_interests = courses_dataset['Interests'].unique().tolist()
        unique_career_goals = courses_dataset['Career Goals'].unique().tolist()

        return courses_dataset, unique_interests, unique_career_goals
    
    courses_path = r"/Users/vinayvarma/Desktop/streamlit-app/data/courses_dataset.csv" 
    comparison_path = r"/Users/vinayvarma/Desktop/streamlit-app/data/course_comparison.csv"

    courses_dataset, unique_interests, \
        unique_career_goals = load_data(courses_path)
    
    comparison_dataset = pd.read_csv(comparison_path)

    # Dropdown for selecting interests and career goals
    name = st.text_input('Your Name')
    interests = st.selectbox("Select your interests", options=unique_interests)
    career_goals = st.selectbox("Select your career goals", options=unique_career_goals)
    resume = st.file_uploader("Upload your resume", type=['pdf'])

    # Button to make prediction
    if st.button('Recommend Courses'):
        if not resume:
            st.error("Please Upoload the resume to proceed!")
            time.sleep(2)
            st.rerun()
        with st.spinner('Recommendation system churning'):
            recommendations, cosine_values = recommend_courses(interests, career_goals, courses_dataset)
            if len(recommendations[0].split(",")) > 1:
                recommendations = recommendations[0].split(",")
                recommendations[1] = recommendations[1].strip()
            st.write("Recommended Courses:")
            for i, course in enumerate(recommendations):
                cosine_row = pd.DataFrame([[name, cosine_values[i], recommendations[i]]], columns=['Name', 'Confidence', 'Course'])
                header = not os.path.exists('course_confidence.csv')
                cosine_row.to_csv('course_confidence.csv', mode='a', header=header, index=False)
                st.markdown(f"<div style='background-color:#ccffcc; font-size:20px; font-weight:bold; border-radius:5px; padding:10px;'>{course}</div><br>", unsafe_allow_html=True)
            st.dataframe(comparison_dataset.loc[(comparison_dataset['Program'] == recommendations[0]) | (comparison_dataset['Program'] == recommendations[1])])

    st.markdown(f'## Feedback Form')
    with st.form('feedback_form'):
        country = st.text_input('Your Country')
        mail = st.text_input('Enter your mail ID')
        course_suggestion = st.text_input('Enter the course you were suggested')
        feedback = st.text_area('Please provide your feedback so we can improve!')
        submitted_feedback = st.form_submit_button('Submit feedback')
        if submitted_feedback:
            resume_filename = resume.name if resume else None
            feedback_row = pd.DataFrame([[name, country, mail, course_suggestion, feedback, resume_filename]], columns=['Name', 'Country', 'Mail', 'course_suggestion', 'Feedback', 'Resume'])
            header = not os.path.exists('feedback.csv')
            feedback_row.to_csv('feedback.csv', mode='a', header=header, index=False)
            # st.success("Thank you for your feedback!")
            if resume:
                with open(os.path.join('uploads', resume.name), "wb") as f:
                    f.write(resume.getbuffer())
            else:
                st.error("Please Upoload teh resume to proceed")
            st.success("Thank you for your feedback!")

def admin():
    import streamlit as st
    import pandas as pd
    import os
    import seaborn as sns
    import matplotlib.pyplot as plt
    import plotly.express as px

    st.markdown("# Admin Page")
    st.write("This page lets the admin assess the feedback provided by users for iterative improvement of the product.")
    # try:
    feedback_df = pd.read_csv('feedback.csv')
    st.markdown("## Resume Review")
    if not feedback_df.empty:
        # Check and update session state for checkboxes
        if 'checkbox_state' not in st.session_state or len(st.session_state.checkbox_state) != len(feedback_df):
            st.session_state.checkbox_state = [False] * len(feedback_df)

        # Create a table with checkboxes and data
        for index, row in feedback_df.iterrows():
            cols = st.columns([1, 3, 1, 5, 4])  # Adjust column widths as necessary
            with cols[0]:
                checkbox = st.checkbox("", key=f"check{index}", value=st.session_state.checkbox_state[index])
                st.session_state.checkbox_state[index] = checkbox
            with cols[1]:
                st.write(f"{row['Name']}")
            with cols[2]:
                st.write(f"{row['Country']}")
            with cols[3]:
                st.write(f"{row['Mail']}")
            with cols[4]:
                st.write(f"{row['Course Suggestion']}")
                if row['Resume']:
                    file_path = os.path.join('uploads', row['Resume'])
                    with open(file_path, "rb") as file:
                        st.download_button(label="Download Resume", data=file.read(), file_name=row['Resume'], mime='application/pdf')
                else:
                    st.write("No resume uploaded")
        
        # Optionally, add a button to process checked items
        if st.button('Process Checked Items'):
            # Processing checked items
            checked_feedback = feedback_df[st.session_state.checkbox_state]
            st.write("Checked items processed:", checked_feedback)
    else:
        st.write("No feedback yet.")
    # except FileNotFoundError:
    #     st.write("No feedback data found.")
    #     st.session_state.checkbox_state = []  # Reset checkbox state if no data is available
    
    st.markdown("## Model Performance Dashboard")

    def plot_admission_confidence(df):
        fig = px.line(df, x='Name', y='Confidence', markers=True, title='Admission Confidence by Candidate')
        fig.update_xaxes(title_text='Candidate Name')
        fig.update_yaxes(title_text='Confidence Level')
        fig.update_traces(textposition='top center')
        st.plotly_chart(fig, use_container_width=True)

    def plot_course_confidence(df):
        fig = px.line(df, x='Name', y='Confidence', color='Course', markers=True, title='Course Recommendation Confidence by Candidate')
        fig.update_xaxes(title_text='Candidate Name')
        fig.update_yaxes(title_text='Confidence Level')
        fig.update_traces(textposition='top center')
        st.plotly_chart(fig, use_container_width=True)

    # Load and plot admission confidence scores
    try:
        admission_df = pd.read_csv('admission_confidence.csv')
        # st.markdown("## Admission Confidence Scores")
        if not admission_df.empty:
            plot_admission_confidence(admission_df)
        else:
            st.write("No admission data yet.")
    except FileNotFoundError:
        st.write("No admission data found.")

    # Load and plot course confidence scores
    try:
        course_df = pd.read_csv('course_confidence.csv')
        if not course_df.empty:
            plot_course_confidence(course_df)
        else:
            st.write("No course data yet.")
    except FileNotFoundError:
        st.write("No course data found.")


page_names_to_funcs = {
    "Menu": intro,
    "Course Recommender": course_recommendation,
    "Admission Checker": admission_selection,
    "Admin Page": admin
}

demo_name = st.sidebar.selectbox("Choose a demo", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()
