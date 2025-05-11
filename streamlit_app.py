import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set page title
st.set_page_config(page_title='MBTI Personality Predictor', layout='wide')

# API endpoint
API_URL = 'http://localhost:5000/predict'

# Title and description
st.title('MBTI Personality Type Predictor')
st.write("""
This app predicts your Myers-Briggs Type Indicator (MBTI) personality type based on your writing style.
Enter up to 3 social media posts, captions, or comments, and our AI model will analyze them to determine your likely MBTI type.
""")

# Create three text input fields
st.subheader('Enter your text samples:')
text1 = st.text_area('Text Sample 1', height=100, placeholder='Enter your first text sample here...')
text2 = st.text_area('Text Sample 2', height=100, placeholder='Enter your second text sample here...')
text3 = st.text_area('Text Sample 3', height=100, placeholder='Enter your third text sample here...')

# Create a button to trigger the prediction
if st.button('Predict My MBTI Type'):
    # Collect non-empty texts
    texts = [t for t in [text1, text2, text3] if t.strip()]
    
    if not texts:
        st.error('Please enter at least one text sample.')
    else:
        # Show a spinner while predicting
        with st.spinner('Analyzing your writing style...'):
            try:
                # Make API request
                response = requests.post(API_URL, json={'texts': texts})
                
                if response.status_code == 200:
                    result = response.json()
                    predicted_type = result['predicted_type']
                    probabilities = result['probabilities']
                    print(probabilities)
                    # Display the predicted type
                    st.success(f'Your predicted MBTI type is: **{predicted_type}**')
                    
                    # Create a layout with columns
                    col1, col2 = st.columns([2, 3])
                    
                    with col1:
                        st.subheader('What does this mean?')
                        
                        # Descriptions for each MBTI dimension
                        descriptions = {
                            'E': 'Extraversion: You gain energy from social interactions',
                            'I': 'Introversion: You gain energy from solitary time',
                            'S': 'Sensing: You focus on concrete facts and details',
                            'N': 'Intuition: You focus on patterns and possibilities',
                            'T': 'Thinking: You make decisions based on logic and analysis',
                            'F': 'Feeling: You make decisions based on values and harmony',
                            'J': 'Judging: You prefer structure and organization',
                            'P': 'Perceiving: You prefer flexibility and spontaneity'
                        }
                        
                        # Show what each letter means
                        for letter in predicted_type:
                            st.write(f"**{letter}**: {descriptions[letter]}")
                    
                    with col2:
                        st.subheader('Top 5 Most Likely Types')
                        
                        # Convert probabilities to DataFrame for display
                        sorted_probabilities = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
                        top_types = list(sorted_probabilities)[:5]
                        df = pd.DataFrame(top_types, columns=['MBTI Type', 'Probability'])
                        
                        # Create a bar chart
                        fig, ax = plt.subplots(figsize=(10, 5))
                        sns.barplot(x='MBTI Type', y='Probability', data=df, ax=ax)
                        ax.set_ylim(0, 1)
                        
                        # Add percentage labels on top of bars
                        for i, v in enumerate(df['Probability']):
                            ax.text(i, v + 0.02, f"{v:.1%}", ha='center')
                        
                        st.pyplot(fig)
                        
                        # Show the probability table
                        df['Probability'] = df['Probability'].apply(lambda x: f"{x:.1%}")
                        st.table(df)
                else:
                    st.error(f"Error: {response.json().get('error', 'Unknown error')}")
            
            except Exception as e:
                st.error(f"Error connecting to the API: {str(e)}")
                st.info("Make sure the Flask API is running at the specified URL.")

# Add some information about MBTI
st.markdown("""
---
### About MBTI

The Myers-Briggs Type Indicator (MBTI) is a personality assessment that categorizes people into 16 different types based on four dimensions:

1. **Extraversion (E) vs. Introversion (I)**: Where you get your energy
2. **Sensing (S) vs. Intuition (N)**: How you gather information
3. **Thinking (T) vs. Feeling (F)**: How you make decisions
4. **Judging (J) vs. Perceiving (P)**: How you organize your life

This app uses a BERT-based deep learning model trained on text samples to predict your likely MBTI type based on your writing style.

**Note**: This is for educational purposes only and should not be considered a professional personality assessment.
""")

# Instructions to run the Streamlit app
st.sidebar.header("How to Run This App")
st.sidebar.code("""
# Install Streamlit
pip install streamlit

# Run the app
streamlit run streamlit_app.py
""")