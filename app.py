import streamlit as st
import pandas as pd
from gensim import corpora, models
import re
import numpy as np 
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords

# --------------------------------------------------
# DOWNLOAD NLTK RESOURCES (CACHED)
# --------------------------------------------------
@st.cache_resource
def load_nltk():
    nltk.download('stopwords')

load_nltk()

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="The Public Pulse",
    page_icon="📊",
    layout="wide"
)

# --------------------------------------------------
# LOAD EXTERNAL CSS
# --------------------------------------------------
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("styles.css")

# --------------------------------------------------
# LOAD DATA (FIXED TIMEZONE ISSUE)
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("tate_piers_comments.csv")
    df['Comment'] = df['Comment'].astype(str)

    # Convert to datetime with UTC, then remove timezone
    df['Published At'] = pd.to_datetime(
        df['Published At'], errors='coerce', utc=True
    )
    df['Published At'] = df['Published At'].dt.tz_localize(None)

    # Keep Date as datetime (not string)
    df['Date'] = df['Published At'].dt.normalize()

    return df

df = load_data()

# --------------------------------------------------
# DATE FILTER (CORRECT & WORKING)
# --------------------------------------------------
st.sidebar.markdown("### 📅 Filter by Date")

start_date = st.sidebar.date_input(
    "Start Date", df['Date'].min().date()
)
end_date = st.sidebar.date_input(
    "End Date", df['Date'].max().date()
)

start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

# --------------------------------------------------
# ANALYSIS (CACHED)
# --------------------------------------------------
@st.cache_data
def analyze_comments(df):
    df = df.copy()

    # Sentiment
    df['Sentiment'] = df['Comment'].apply(
        lambda x: TextBlob(x).sentiment.polarity
    )

    def sentiment_label(score):
        if score > 0:
            return "Positive"
        elif score < 0:
            return "Negative"
        else:
            return "Neutral"

    df['Sentiment Category'] = df['Sentiment'].apply(sentiment_label)

    # Toxicity
    toxic_words = ['hate', 'kill', 'idiot', 'stupid', 'racist', 'terrorist']
    df['Toxic'] = df['Comment'].apply(
        lambda x: any(word in x.lower() for word in toxic_words)
    )

    return df

df = analyze_comments(df)
# ---------------- LDA FUNCTION ----------------
@st.cache_data
def run_lda(df, num_topics=5):

    def clean_text(text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text.split()

    texts = df['Comment'].dropna().apply(clean_text)

    stop_words = set(stopwords.words('english'))
    texts = [[word for word in doc if word not in stop_words] for doc in texts]

    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    lda_model = models.LdaModel(
        corpus,
        num_topics=num_topics,
        id2word=dictionary,
        passes=10
    )

    topics = lda_model.print_topics(num_words=5)

    return topics

def get_topic_names(lda_model, num_words=3):
    topic_names = {}

    for i, topic in lda_model.show_topics(num_topics=-1, num_words=num_words, formatted=False):
        words = [word for word, prob in topic]
        
        # Create readable name
        name = " ".join(words[:3]).title()
        
        topic_names[i] = name

    return topic_names

# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------
st.sidebar.markdown("""
<style>
div[role="radiogroup"] > label {
    margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)

section = st.sidebar.radio(
    "",
    [
        "Dashboard",
        "Dataset Overview",
        "Sentiment Analysis",
        "Sentiment Over Time",
        "Topic Modeling",
        "Toxicity Detection",
        "Live Comment Analyzer",
        "Download Results"
    ]
)

# --------------------------------------------------
# WORDCLOUD CACHE
# --------------------------------------------------
@st.cache_data
def generate_wordcloud(text):
    return WordCloud(
        width=1200,
        height=500,
        background_color="black",
        colormap="cool",
        stopwords=set(stopwords.words('english'))
    ).generate(text)

from gensim import corpora
from gensim.models import LdaModel
import re

@st.cache_data
def run_lda(df, num_topics=5):
    texts = df['Comment'].dropna().tolist()

    # Clean text
    cleaned = [
        re.sub(r'[^a-zA-Z ]', '', text.lower()).split()
        for text in texts
    ]

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    cleaned = [[word for word in doc if word not in stop_words] for doc in cleaned]

    # Create dictionary & corpus
    dictionary = corpora.Dictionary(cleaned)
    corpus = [dictionary.doc2bow(text) for text in cleaned]

    # Train LDA model
    lda_model = LdaModel(
        corpus,
        num_topics=num_topics,
        id2word=dictionary,
        passes=10
    )

    # Extract topics
    topics = lda_model.print_topics(num_words=6)

    return topics
# --------------------------------------------------
# SECTION HELPERS
# --------------------------------------------------
def section_container(title):
    st.markdown(
        f"<div class='section-box'><h1 class='animated-title'>{title}</h1>",
        unsafe_allow_html=True
    )

def end_section():
    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------
if section == "Dashboard":
    st.markdown("""
    <div class="hero">
        <img src="https://media.giphy.com/media/3oKIPEqDGUULpEU0aQ/giphy.gif" width="140">
        <h1>The Public Pulse</h1>
        <p>Visual Intelligence for Public Opinion Analysis</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card-grid">
        <div class="card">
            <img src="https://media.giphy.com/media/l0HlSNOxJB956qwfK/giphy.gif" width="115" height="150">
            <h1>Sentiment Analysis</h1>
            <p>Measures emotional polarity of comments.</p>
        </div>
        <div class="card">
            <img src="https://media.giphy.com/media/xT9IgzoKnwFNmISR8I/giphy.gif" width="160">
            <h2>Topic Modeling</h2>
            <p>Extracts dominant discussion themes.</p>
        </div>
        <div class="card">
            <img src="https://media.giphy.com/media/26tn33aiTi1jkl6H6/giphy.gif">
            <h1>Toxicity Detection</h1>
            <p>Identifies abusive & harmful speech.</p>
        </div>
        <div class="card">
            <img src="https://media.giphy.com/media/3o7qE1YN7aBOFPRw8E/giphy.gif">
            <h1>Live Analyzer</h1>
            <p>Instant analysis of new comments.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    total_comments = df.shape[0]
    positive_pct = (df['Sentiment Category'] == "Positive").mean() * 100
    toxic_pct = (df['Toxic'] == True).mean() * 100

    st.markdown("""
    <div class="section-box">
        <h2 class="animated-title"> Key Insights</h2>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Comments", total_comments)
    c2.metric("Positive Sentiment (%)", f"{positive_pct:.2f}%")
    c3.metric("Toxic Content (%)", f"{toxic_pct:.2f}%")
    c4.metric("Platform", "YouTube")

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------
# DATASET OVERVIEW
# --------------------------------------------------
elif section == "Dataset Overview":
    section_container("📁 Dataset Overview")
    st.dataframe(df.head(20))
    st.success(f"Total Comments: {df.shape[0]}")
    end_section()

# --------------------------------------------------
# SENTIMENT ANALYSIS
# --------------------------------------------------
elif section == "Sentiment Analysis":
    section_container("😊 Sentiment Analysis")

    counts = df['Sentiment Category'].value_counts()

    # 🔥 Dynamic chart toggle
    chart_type = st.radio("Select Chart Type", ["Bar Chart", "Pie Chart"])

    if chart_type == "Bar Chart":
        fig = px.bar(
            x=counts.index,
            y=counts.values,
            labels={'x': 'Sentiment', 'y': 'Count'},
            title="Sentiment Distribution",
            color=counts.index
        )
    else:
        fig = px.pie(
            values=counts.values,
            names=counts.index,
            title="Sentiment Proportion"
        )

    st.plotly_chart(fig, use_container_width=True)
    end_section()

# --------------------------------------------------
# SENTIMENT OVER TIME
# --------------------------------------------------
elif section == "Sentiment Over Time":
    section_container("📈 Sentiment Over Time")

    trend = df.groupby('Date')['Sentiment'].mean().reset_index()

    fig = px.line(
        trend,
        x='Date',
        y='Sentiment',
        title="Average Sentiment Over Time",
        markers=True
    )

    st.plotly_chart(fig, use_container_width=True)
    end_section()

# --------------------------------------------------
# TOPIC MODELING
# --------------------------------------------------
elif section == "Topic Modeling":
    section_container("🧠 Topic Modeling & Trend Analysis")

    # ---------------- WORD CLOUD ----------------
    text_data = " ".join(df['Comment'].dropna())
    wc = generate_wordcloud(text_data)

    fig_wc, ax_wc = plt.subplots(figsize=(12, 5))
    ax_wc.imshow(np.array(wc.to_image()))
    ax_wc.axis("off")

    st.subheader("📌 Discussion Word Cloud")
    st.pyplot(fig_wc)

    # ---------------- TOPIC TRENDS ----------------
    st.subheader("📈 Topic Trends Over Time")

    topic_keywords = {
        "Andrew Tate": ["tate", "andrew"],
        "Piers Morgan": ["piers", "morgan"],
        "Gender & Society": ["man", "woman", "men", "women"],
        "Politics & Conflict": ["israel", "hamas"],
        "Health & COVID": ["covid", "vaccine"]
    }

    trend_df = df[['Date', 'Comment']].copy()

    # Create topic columns
    for topic, keywords in topic_keywords.items():
        trend_df[topic] = trend_df['Comment'].str.lower().apply(
            lambda x: any(word in x for word in keywords)
        )

    # Count topic occurrences per date
    trend_counts = trend_df.groupby('Date')[list(topic_keywords.keys())].sum()
    trend_counts_reset = trend_counts.reset_index()

    # Convert to long format (for animation)
    trend_long = trend_counts_reset.melt(
        id_vars='Date',
        var_name='Topic',
        value_name='Count'
    )

    # ---------------- TOPIC SELECTOR ----------------
    selected_topics = st.multiselect(
        "Select Topics",
        list(topic_keywords.keys()),
        default=list(topic_keywords.keys())
    )

    st.info("💡 Use the play button ▶️ to see how topics evolve over time")

    # ---------------- ANIMATED GRAPH ----------------
    if selected_topics:
        filtered_data = trend_long[trend_long['Topic'].isin(selected_topics)]

        fig = px.bar(
            filtered_data,
            x='Topic',
            y='Count',
            color='Topic',
            animation_frame="Date",
            title="📊 Topic Evolution Over Time"
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("⚠️ Please select at least one topic to display the graph.")

        # ---------------- REAL LDA TOPICS ----------------
    st.subheader("🧠 Discovered Topics (LDA Model)")

    num_topics = st.slider("Select number of topics", 2, 10, 5)

    topics = run_lda(df, num_topics)

    for i, topic in topics:
        st.write(f"**Topic {i+1}:** {topic}")

    end_section()


# --------------------------------------------------
# TOXICITY DETECTION
# --------------------------------------------------
elif section == "Toxicity Detection":
    section_container("⚠️ Toxicity Detection")

    counts = df['Toxic'].value_counts()

    tox_df = pd.DataFrame({
        "Category": ["Non-Toxic", "Toxic"],
        "Count": [counts.get(False, 0), counts.get(True, 0)]
    })

    fig = px.bar(
        tox_df,
        x="Category",
        y="Count",
        color="Category",
        title="Toxic vs Non-Toxic Comments"
    )

    st.plotly_chart(fig, use_container_width=True)
    end_section()

# --------------------------------------------------
# LIVE COMMENT ANALYZER
# --------------------------------------------------
elif section == "Live Comment Analyzer":
    section_container("📝 Live Comment Analyzer")

    custom_negative_words = [
        "narcissist","idiot","stupid","hate","useless","loser",
        "dumb","trash","ugly","pathetic","moron","fraud",
        "scam","garbage","fool","disgusting","toxic","hypocrite",
        "idiot", "stupid", "dumb", "moron", "fool", "loser", "clown",
        "narcissist", "psycho", "crazy", "delusional", "pathetic",
        "useless", "worthless", "trash", "garbage", "scum", "jerk",
        "hate", "disgusting", "awful", "terrible", "horrible",
        "worst", "toxic", "evil", "corrupt", "fake", "fraud",
        "liar", "hypocrite", "shameful", "ridiculous",
        "laughable", "embarrassing", "cringe", "lame",
        "nonsense", "absurd", "ignorant", "illiterate",
        "retard", "retarded", "snowflake", "bootlicker",
        "simp", "clueless", "brainwashed", "triggered",
        "propaganda", "sellout", "manipulative", "biased",
        "dictator", "extremist", "radical",
        "problematic", "questionable", "disturbing",
        "concerning", "unacceptable","no sense"
    ]

    text = st.text_area("Enter a comment", height=200)

    if st.button("Analyze"):

        score = TextBlob(text).sentiment.polarity
        text_lower = text.lower()

        if any(word in text_lower for word in custom_negative_words):
            st.error("Negative 😠")
        elif score > 0:
            st.success("Positive 😊")
        elif score < 0:
            st.error("Negative 😠")
        else:
            st.warning("Neutral 😐")

    end_section()

# --------------------------------------------------
# DOWNLOAD RESULTS
# --------------------------------------------------
elif section == "Download Results":
    section_container("⬇️ Download Results")
    st.download_button(
        "Download CSV",
        df.to_csv(index=False),
        "public_pulse_results.csv",
        "text/csv"
    )
    end_section()

