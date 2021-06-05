import streamlit as st
import pandas as pd
import numpy as np
#from streamlit_folium import folium_static
import warnings
warnings.filterwarnings('ignore')
#import seaborn as sns
#from sklearn.model_selection import train_test_split, cross_val_score
#from sklearn.neighbors import KNeighborsClassifier 
#from sklearn.metrics import accuracy_score,roc_curve, auc, confusion_matrix, classification_report
#from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances, cosine_similarity
#from sklearn.metrics import mean_squared_error, mean_absolute_error
#from sklearn.linear_model import LinearRegression

#import keyring
#import time

#set
st.set_page_config(
    page_title="Group 3:",
    layout="wide",
    initial_sidebar_state="expanded",
    
)

st.markdown("""
<style>
.big-font {
    font-size:70px !important;
    color:#1DB954;
}
.med-font {
    font-size:40px !important;
    color:#1DB954;
}
</style>
""", unsafe_allow_html=True)


# # Sprint 2 Project
# Team Patrisha: Adam, Bianca, Chatty, Louie, Matthew, Shawn

#get_ipython().run_line_magic('matplotlib', 'inline')

my_page = st.sidebar.radio("Contents", ["Recommender Engine", "About the Team"])

if my_page == 'Recommender Engine':
    st.markdown('<p class="big-font">Recommender Engine</p>', unsafe_allow_html=True)

    data_pred = pd.read_csv('data/data_pred_genre_revenue.csv')
    seed_track = pd.read_csv('data/seed_tracks_with_predicted_genre.csv')
 #   album_df = pd.read_csv('data/album_tracks.csv')
 #   album_tracks_df = pd.read_csv('data/album_tracks_data.csv')
 #   chart_tracks_df =pd.read_csv("data/spotify_daily_charts_tracks_predicted_genres.csv")
    
    artist_name = ['Olivia Rodrigo', 'Harry Styles']
    artist_img = ['https://i.scdn.co/image/ab67616d0000b273a91c10fe9472d9bd89802e5a']
    artist_img1=['https://i.scdn.co/image/b2163e7456f3d618a0e2a4e32bc892d6b11ce673']
    
    data_pred = data_pred[~data_pred["artist_name"].isin(artist_name)]
    option = st.sidebar.selectbox('Name of Artist:', artist_name)
    option1 = st.sidebar.selectbox('Seed Track:', seed_track[seed_track['artist_name']==option]['track_name'].unique())
    col1, mid, col2 = st.beta_columns([1,1,5])
    

    if option=='Olivia Rodrigo':
        with col1:
            st.image(artist_img, width=300)
        with col2:
            st.write('<span style="color: #FFFFFF; font-size: 45px;"><b>Olivia Rodrigo</b></span>',unsafe_allow_html=True)
            st.write('<span style="color: #FFFFFF; font-size: 15px;"><b>Olivia Isabel Rodrigo is an American actress, singer, and songwriter. She is known for her roles as Paige Olvera on the Disney Channel series Bizaardvark and Nini Salazar-Roberts on the Disney+ series High School Musical: The Musical: The Series.</b></span>',unsafe_allow_html=True)
            
            if option1=='drivers license':
                my_expander = st.beta_expander('See Playlist with Similar Tracks')
                my_expander1 = st.beta_expander('See Playlist with Dissimilar Tracks')
                with my_expander:
                       st.write('<table>'
                       '<tr>'
                   '<td><iframe src="https://open.spotify.com/embed/playlist/6eb2shyVSE998j0E6fZagI" width="100%" height="380" frameBorder="0" allowtransparency="true" allow="encrypted-media"></iframe></td>'
                   '</tr>'
                   '</table>', unsafe_allow_html=True) 
                with my_expander1:
                       st.write('<table>'
                       '<tr>'
                   '<td><iframe src="https://open.spotify.com/embed/playlist/1kkMoUYXMHs8x0ObUCst1z" width="100%" height="380" frameBorder="0" allowtransparency="true" allow="encrypted-media"></iframe></td>'
                   '</tr>'
                   '</table>', unsafe_allow_html=True) 
        
    else:
        with col1:
            st.image(artist_img1, width=300)
        with col2:
            st.write('<span style="color: #FFFFFF; font-size: 45px;"><b>Harry Styles</b></span>',unsafe_allow_html=True)
            st.write('<span style="color: #FFFFFF; font-size: 15px;"><b>English singer Harry Styles has released two studio albums, one extended play (EP), one video album, nine singles, eight music videos and one promotional single. Styles\'s music career began in 2010 as a member of the boy band One Direction.</b></span>',unsafe_allow_html=True)
            
            
            if option1=='Watermelon Sugar':
                my_expander2 = st.beta_expander('See Playlist with Similar Tracks')
                my_expander3 = st.beta_expander('See Playlist with Dissimilar Tracks')
                with my_expander2:
                       st.write('<table>'
                       '<tr>'
                   '<td><iframe src="https://open.spotify.com/embed/playlist/6hNBkzj1fwKJ4th8WMfdve" width="100%" height="380" frameBorder="0" allowtransparency="true" allow="encrypted-media"></iframe></td>'
                   '</tr>'
                   '</table>', unsafe_allow_html=True) 
                with my_expander3:
                       st.write('<table>'
                       '<tr>'
                   '<td><iframe src="https://open.spotify.com/embed/playlist/7cDlVjMXCNNNYlvWte22kG" width="100%" height="380" frameBorder="0" allowtransparency="true" allow="encrypted-media"></iframe></td>'
                   '</tr>'
                   '</table>', unsafe_allow_html=True) 
        
    
    seed_track_data = seed_track[seed_track['track_name']==option1].iloc[0]
    
    feature_cols = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness','instrumentalness','liveness', 'valence', 'tempo']
    data_pred['cosine_dist'] = data_pred.apply(lambda x: 1-cosine_similarity(x[feature_cols].values.reshape(1, -1),\
                                                                    seed_track_data[feature_cols].values.reshape(1, -1))\
                                                                     .flatten()[0], axis=1)
    
    print('Recommendations for ', option1, ': Most Similar, by cosine distance')
    recommendation_df = data_pred[data_pred['track_id']!=seed_track_data['track_id']].sort_values('cosine_dist')[['track_name','artist_name','cosine_dist','predicted_genre', 'pred_position_class', 'potential_revenue', 'track_id']][:10]
    not_similar_df = data_pred[data_pred['track_id']!=seed_track_data['track_id']].sort_values('cosine_dist', ascending=False)[['track_name','artist_name','cosine_dist','predicted_genre', 'pred_position_class', 'potential_revenue', 'track_id']][:10]
    
    potential_rank = {1: '1-10', 2:'11-20', 3:'21-30', 4:'31-40', 5:'41-50', 6:'51-60', 7:'61-70', 8:'71-80', 9:'81-90', 10:'91-100',
                     11: '101-110', 12:'111-120', 13:'121-130', 14:'131-140', 15:'141-150', 16:'151-160', 17:'161-170', 18:'171-180', 19:'181-190', 20:'191-200'} 
    recommendation_df['pred_position_class'] = recommendation_df['pred_position_class'].map(potential_rank)
    not_similar_df['pred_position_class'] = not_similar_df['pred_position_class'].map(potential_rank)
    renamed_columns = ['Song Name', 'Artist', 'Similarity Score', 'Genre', 'Potential Rank', 'Daily Income ($)', 'ID']
    recommendation_df.columns = renamed_columns
    not_similar_df.columns = renamed_columns
    st.markdown('<p class="med-font">Top 10 Recommended Songs</p>', unsafe_allow_html=True)
    st.write(recommendation_df)
    st.markdown('<p class="med-font">Top 10 Not Similar Songs</p>', unsafe_allow_html=True)
    st.write(not_similar_df)
    
elif my_page == 'About the Team':
    st.markdown('<p class="big-font">About the Team</p>', unsafe_allow_html=True)
    st.write('<span style="color: red; font-size: 20px;"><b>Mentor</b></span>',unsafe_allow_html=True)
    st.write('<span style="color: #FFFFFF; font-size: 15px;"><b>Patricia</b></span>',unsafe_allow_html=True)
    st.write('<span style="color: red; font-size: 20px;"><b>Team</b></span>',unsafe_allow_html=True)
    st.write('<span style="color: #FFFFFF; font-size: 15px;"><b>Adam</b></span>',unsafe_allow_html=True)
    st.write('<span style="color: #FFFFFF; font-size: 15px;"><b>Bianca</b></span>',unsafe_allow_html=True)
    st.write('<span style="color: #FFFFFF; font-size: 15px;"><b>Charity</b></span>',unsafe_allow_html=True)
    st.write('<span style="color: #FFFFFF; font-size: 15px;"><b>Louie</b></span>',unsafe_allow_html=True)
    st.write('<span style="color: #FFFFFF; font-size: 15px;"><b>Matthew</b></span>',unsafe_allow_html=True)
    st.write('<span style="color: #FFFFFF; font-size: 15px;"><b>Shawn</b></span>',unsafe_allow_html=True)
