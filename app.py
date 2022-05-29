import json
import requests
import streamlit as st
import streamlit.components.v1 as stc
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from streamlit_lottie import st_lottie
from streamlit_lottie import st_lottie_spinner


# LOAD DATASET
def load_dataset(data):
    df = pd.read_csv(data)
    return df


def vectorize_t2c(data):
    count_vector = CountVectorizer()
    cv_mat = count_vector.fit_transform(data)
    # GET THE COSINE
    cosine_sim = cosine_similarity(cv_mat)
    print(cosine_sim)
    return cosine_sim

@st.cache
def recommend_cars(title, cosine_sim, df, no_ofRecommendations=5):
    # indices of the course
    car_indices = pd.Series(df.index, index=df['Car']).drop_duplicates()
    # Index of course
    idx = car_indices[title]

    # Look into the cosine matr for that index
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    selected_car_indices = [i[0] for i in sim_scores[1:]]
    selected_car_scores = [i[0] for i in sim_scores[1:]]

    # Get the dataframe & title
    result_df = df.iloc[selected_car_indices]
    result_df['similarity_score'] = selected_car_scores
    final_recommended_cars = result_df[['Car', 'Style', 'VehicleType', 'PriceRange', 'Transmission', 'similarity_score']]
    return final_recommended_cars.head(no_ofRecommendations)

RESULT_TEMP = """
<div style="width:90%;height:100%;margin:1px;padding:5px;position:relative;border-radius:5px;border-bottom-right-radius: 60px;
box-shadow:0 0 5px 5px #ccc; background-color: #99E7F1;
  border-left: 5px solid #30a4c7;">
<h4>{}</h4>
<p style="color:blue;"><span style="color:black;">CAR STYLE: </span>{}</p>
<p style="color:blue;"><span style="color:black;">VEHICLE TYPE: </span>{}</p>
<p style="color:blue;"><span style="color:black;">PRICE RANGE: </span>{}</p>
<p style="color:blue;"><span style="color:black;">TRANSMISSION TYPE: </span>{}</p>
<p style="color:blue;"><span style="color:black;">SIMILARITY SCORE: </span>{}</p>
</div>
"""

# Search For CARS
@st.cache
def search_car_if_not_found(term, df):
    result_df = df[df['Car'].str.contains(term)]
    return result_df

#UPLOAD URL OF ANIMATION AS A JSON FILE
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def main():
    st.title("CAR RECOMMENDATION WEBAPP")
    menu = ['HOME', 'RECOMMEND CARS']
    choice = st.sidebar.selectbox("MENU", menu)
    df = load_dataset("G:/CourseWebapp/data/newCarsIndia.csv")
    if choice == 'HOME':
        hello_ani = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_1pxqjqps.json")
        st_lottie(hello_ani, speed=1, height=500, width=500, quality="medium", loop=True)
        st.write("WE RECOMMEND YOU CARS ON THE BASIS OF A SAMPLE MODEL YOU PREFER TO LOOK FOR.")
        st.warning("CHOOSE 'RECOMMEND CARS' FROM THE SELECTBOX ON THE SIDE BAR TO SEE RECOMMENDATIONS!")
    else:
        cosine_sim_mat = vectorize_t2c(df['Car'])
        search = st.text_input("SEARCH CAR")
        no_ofRecommendations = st.sidebar.number_input("NUMBER", 4, 20, 7)

        if st.button("RECOMMEND"):

            if search is not None:
                try:
                    results = recommend_cars(search, cosine_sim_mat, df, no_ofRecommendations)
                    with st.expander("VIEW RECOMMENDATIONS"):
                        results_json = results.to_dict('index')

                        for row in results.iterrows():
                            rec_car = row[1][0]
                            rec_style = row[1][1]
                            rec_vehicleType = row[1][2]
                            rec_price = row[1][3]
                            rec_transmission = row[1][4]
                            rec_score = row[1][5]

                            st.write(rec_car)
                            stc.html(RESULT_TEMP.format(rec_car, rec_style, rec_vehicleType, rec_price, rec_transmission, rec_score), height=350)

                except:
                    lottie_hello = load_lottieurl("https://assets5.lottiefiles.com/private_files/lf30_fn9xcfqg.json")
                    st_lottie(lottie_hello, speed=1, height=400, width=400, quality="medium", loop=True)
                    st.warning("RESULTS NOT FOUND")
                    st.warning("MODEL NAME SPELLED INCORECTLY! (TRY WRITING EVERY FIRST LETTER OF WORD IN CAPITAL)")
                # st.write(result)



if __name__ == '__main__':
    main()
