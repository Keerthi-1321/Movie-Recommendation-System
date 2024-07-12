import streamlit as st
import pandas as pd
import mysql.connector
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import base64  # Required for encoding the CSV data

# Function to establish MySQL connection
def connect_to_db():
    return mysql.connector.connect(
        host="localhost",  # Your host, usually localhost
        user="root",       # Your username (assuming root with no password)
        database="movie_database"  # Your database name
    )

# Function to fetch data from MySQL for the movie database page
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
            st.success("Movie details saved successfully!")
            cursor.close()
            conn.close()
    except mysql.connector.Error as e:
        st.error(f"Error: {e}")

# Function to download data as CSV
def download_csv(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # Encoding to b64
    href = f'<a href="data:file/csv;base64,{b64}" download="movie_database.csv">Download CSV File</a>'
    return href

# Load the dataset for movie recommendation system
def load_data_for_recommendation():
    df = pd.read_excel('movie_review.xlsx')
    df.fillna('', inplace=True)
    df['combined_features'] = df['director_name'] + ' ' + df['actor_1_name'] + ' ' + df['actor_2_name'] + ' ' + df['actor_3_name'] + ' ' + df['genres']
    return df

# Initialize TF-IDF Vectorizer for recommendation system
def initialize_tfidf(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined_features'].values.astype('U'))
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

# Function to get movie recommendations
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

# Streamlit App
def main():
    st.title('Movie Recommendation and Database Management')

    # Sidebar navigation
    page = st.sidebar.selectbox("Choose a page", ["Movie Recommendation System", "Movie Database"])

    if page == "Movie Recommendation System":
        st.header('Movie Recommendation System')
        st.write('Enter a movie title to get recommendations.')

        # Load data and initialize TF-IDF for recommendation system
        df_recommendation = load_data_for_recommendation()
        cosine_sim = initialize_tfidf(df_recommendation)

        # Input movie title
        movie_title = st.text_input('Enter a movie title')

        # Search button
        if st.button('Search'):
            if movie_title:
                recommendations = get_recommendations(movie_title, df_recommendation, cosine_sim)
                if not recommendations.empty:
                    st.subheader('Top 10 Similar Movies:')
                    st.dataframe(recommendations, use_container_width=True)  # Display recommendations fitting the container width
                else:
                    st.write('No recommendations found.')
            else:
                st.write('Please enter a movie title and click Search.')

    elif page == "Movie Database":
        st.header('Movie Database')
        st.write('Here you can view and manage the movie database.')

        # Pagination settings
        page_size = 10
        page_number = st.number_input('Enter page number:', min_value=1, value=1)

        # Calculate offset based on page number
        offset = (page_number - 1) * page_size

        # Fetch data from MySQL for movie database page
        df_movie_database = fetch_data_from_mysql(offset, page_size)

        if df_movie_database is not None and not df_movie_database.empty:
            st.write('Movie Database:')
            st.dataframe(df_movie_database, use_container_width=True)  # Display the fetched data fitting the container width

            # Download CSV button
            st.markdown(download_csv(df_movie_database), unsafe_allow_html=True)

            # Pagination controls
            st.markdown('---')
            
            # Add Movie section
            st.write('Add a New Movie:')
            movie_title = st.text_input('Movie Title')
            director_name = st.text_input('Director Name')
            actor_1_name = st.text_input('Actor 1 Name')
            actor_2_name = st.text_input('Actor 2 Name')
            actor_3_name = st.text_input('Actor 3 Name')
            genres = st.text_input('Genres')

            # Save button for adding a new movie
            if st.button('Save'):
                if movie_title and director_name and genres:  # Basic validation
                    insert_data_to_mysql(movie_title, director_name, actor_1_name, actor_2_name, actor_3_name, genres)
                    st.success('Movie details saved successfully!')
                    # Refresh the page to display the updated data after saving
                    st.experimental_rerun()
                else:
                    st.warning('Please enter at least Movie Title, Director Name, and Genres.')

        else:
            st.write('No records found.')

if __name__ == "__main__":
    main()
