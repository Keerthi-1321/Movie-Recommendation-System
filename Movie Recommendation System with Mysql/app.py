import streamlit as st
import pandas as pd
import mysql.connector
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import base64
import os

# Function to establish MySQL connection
def connect_to_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        database="movie_database"
    )

# Function to fetch data from MySQL
def fetch_data_from_mysql(offset, limit):
    try:
        conn = connect_to_db()
        if conn.is_connected():
            cursor = conn.cursor(dictionary=True)
            query = f"SELECT * FROM movies LIMIT {limit} OFFSET {offset}"
            cursor.execute(query)
            data = cursor.fetchall()
            cursor.close()
            conn.close()
            return pd.DataFrame(data)
    except mysql.connector.Error as e:
        st.error(f"Error: {e}")
        return None

# Function to insert data into MySQL
def insert_data_to_mysql(movie_title, director_name, actor_1_name, actor_2_name, actor_3_name, genres):
    try:
        conn = connect_to_db()
        if conn.is_connected():
            cursor = conn.cursor()
            query = "INSERT INTO movies (movie_title, director_name, actor_1_name, actor_2_name, actor_3_name, genres) VALUES (%s, %s, %s, %s, %s, %s)"
            cursor.execute(query, (movie_title, director_name, actor_1_name, actor_2_name, actor_3_name, genres))
            conn.commit()
            cursor.close()
            conn.close()
    except mysql.connector.Error as e:
        st.error(f"Error: {e}")

# Function to download CSV
def download_csv(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="movie_database.csv">Download CSV File</a>'
    return href

# Load dataset for recommendation system
def load_data_for_recommendation():
    df = pd.read_excel('movie_review.xlsx')
    df.fillna('', inplace=True)
    df['combined_features'] = df['director_name'] + ' ' + df['actor_1_name'] + ' ' + df['actor_2_name'] + ' ' + df['actor_3_name'] + ' ' + df['genres']
    return df

# Initialize TF-IDF
def initialize_tfidf(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined_features'].values.astype('U'))
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

# Get recommendations
def get_recommendations(title, df, cosine_sim):
    idx = df.index[df['movie_title'].str.lower() == title.lower()].tolist()
    if not idx:
        return pd.DataFrame()
    idx = idx[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return df.iloc[movie_indices]

# Streamlit app
def main():
    st.set_page_config(page_title='Movie Management & Recommendation System', page_icon="🎬", layout="wide")
    
    # Inject CSS for sidebar background only
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {
            background-image: url('https://wallpapercave.com/wp/wp1945898.jpg');  /* Change to your background image URL */
            background-size: cover;
            width: 300px;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
            padding: 0;
        margin: 0;
        }
         .centered-title {
            text-align: center;
            font-size: 40px;
        }
        </style>
        """, unsafe_allow_html=True
    )

    st.title('🎥 Movie Management & Recommendation System')

    # Sidebar navigation with default text and dropdown options
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Choose a page", "🎬 Movie Recommendation System", "📋 Movie Database"]
    )

    if page == "Choose a page":
        st.markdown('---')
        st.write('## Project Overview')

        st.write('1. **Dual Functionality:** The application integrates two main functionalities: a movie recommendation system and a movie database management tool. Users can receive movie recommendations based on their preferences and manage movie records in a database.')
        st.write('2. **Movie Recommendation System:** Leveraging machine learning techniques, the recommendation system uses TF-IDF vectorization and cosine similarity to provide personalized movie suggestions. Users can input a movie title and receive a list of similar movies based on the content and features of the provided movie.')
        st.write('3. **Database Management:** The system allows users to view, manage, and update a movie database. It connects to a MySQL database to fetch movie records, add new movies, and save changes. Users can also download the database as a CSV file for offline access.')
        st.write('4. **User-Friendly Interface:** The application features a clean and intuitive interface built with Streamlit. It includes a sidebar for navigation, where users can switch between the movie recommendation and database management pages. The interface is designed for ease of use and provides real-time feedback.')
        st.write('5. **Visual and Functional Customization:** The application includes a customized sidebar with a background image and provides a clear layout for displaying recommendations and database records. It offers functionalities like pagination for database records and the ability to add new movies through a form.')
    
    if page == "🎬 Movie Recommendation System":
        st.markdown("<h1 class='centered-title'>🎬 Movie Recommendation System</h1>", unsafe_allow_html=True)
        
        # Load data and initialize TF-IDF
        df_recommendation = load_data_for_recommendation()
        cosine_sim = initialize_tfidf(df_recommendation)

        # Input movie title
        movie_title = st.text_input('Enter a movie title to get recommendations:')

        # Search button
        if st.button('Search'):
            if movie_title:
                recommendations = get_recommendations(movie_title, df_recommendation, cosine_sim)
                if not recommendations.empty:
                    st.subheader('Top 10 Similar Movies:')
                    st.dataframe(recommendations, use_container_width=True)
                else:
                    st.write('No recommendations found.')
            else:
                st.write('Please enter a movie title and click Search.')

    elif page == "📋 Movie Database":
        st.markdown("<h1 class='centered-title'>📋 Movie Database</h1>", unsafe_allow_html=True)
        st.write('You can view and manage the Movie Database here.')

        # Pagination
        page_size = 10
        page_number = st.number_input('Enter page number:', min_value=1, value=1)

        # Fetch data from MySQL
        offset = (page_number - 1) * page_size
        df_movie_database = fetch_data_from_mysql(offset, page_size)

        if df_movie_database is not None and not df_movie_database.empty:
            st.write('Movie Database:')
            st.dataframe(df_movie_database, use_container_width=True)

            # Download CSV button
            st.markdown(download_csv(df_movie_database), unsafe_allow_html=True)

            # Add a new movie
            st.markdown('---')
            st.write('Add a New Movie:')
            movie_title = st.text_input('Movie Title')
            director_name = st.text_input('Director Name')
            actor_1_name = st.text_input('Actor 1 Name')
            actor_2_name = st.text_input('Actor 2 Name')
            actor_3_name = st.text_input('Actor 3 Name')
            genres = st.text_input('Genres')

            if st.button('Save'):
                if movie_title and director_name and genres:
                    insert_data_to_mysql(movie_title, director_name, actor_1_name, actor_2_name, actor_3_name, genres)
                    st.success(f"Movie '{movie_title}' has been successfully added to the database!")  # Display success message
                    st.experimental_rerun()
                else:
                    st.warning('Please enter Movie Title, Director Name, and Genres.')

if __name__ == "__main__":
    main()
