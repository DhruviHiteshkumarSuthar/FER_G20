from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import cv2
import base64
from tensorflow.keras.models import load_model
import numpy as np
from collections import deque
import atexit

# Load the pre-trained emotion recognition model
try:
    model = load_model('emotion_model.h5')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Global Variables
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Camera could not be accessed.")
    exit(1)

predicted_emotions = deque(maxlen=100)
timestamps = deque(maxlen=100)
emotion_trends = {label: deque(maxlen=100) for label in emotion_labels}


# Function to process video frames
def process_frame():
    ret, frame = cap.read()
    if not ret:
        return None, None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(gray, (48, 48)) / 255.0
    face = np.expand_dims(face, axis=(0, -1))
    prediction = model.predict(face, verbose=0)
    emotion = emotion_labels[np.argmax(prediction)]
    cv2.putText(frame, emotion, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    return frame, emotion


def encode_frame_to_base64(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')


# Release camera resources on shutdown
def shutdown():
    cap.release()
    cv2.destroyAllWindows()


atexit.register(shutdown)

# Dash App Initialization
app = Dash(__name__, external_stylesheets=[dbc.themes.LUX])
server = app.server

# Layout Design
app.layout = dbc.Container([
    # Header
    dbc.NavbarSimple(
    brand=[
        html.Img(src='/assets/logo.png', style={
            'height': '50px', 'width': '50px', 'borderRadius': '50%', 'marginRight': '10px'}) , 
        "Emotion Insights"
    ],
    color="primary",
    dark=True,
    className="mb-4"
),


    # Main Content
    dbc.Row([
        # Video Feed Column
        dbc.Col([
            html.Div([
                html.H5("Live Video Feed", className="text-center mb-3"),
                html.Img(id='video-feed', style={'width': '100%', 'borderRadius': '10px'})
            ], className="p-4 bg-white rounded shadow-sm"),
        ], width=6),

        # Metrics Column
        dbc.Col([
            html.Div([
                html.H5("Current Emotion", className="text-center mb-3"),
                html.H2(id='current-emotion', className="text-center text-primary"),
                dcc.Graph(id='emotion-heatmap', config={'displayModeBar': False}),
            ], className="p-4 bg-white rounded shadow-sm"),
        ], width=6),
    ], className="mb-4"),

    # Graphs Row
    dbc.Row([
        dbc.Col([
            html.Div([
                dcc.Graph(id='emotion-line-chart', config={'displayModeBar': False}),
            ], className="p-4 bg-white rounded shadow-sm"),
        ], width=6),

        dbc.Col([
            html.Div([
                dcc.Graph(id='emotion-pie-chart', config={'displayModeBar': False}),
            ], className="p-4 bg-white rounded shadow-sm"),
        ], width=6),
    ]),

    # Interval for Live Updates
    dcc.Interval(id='update-interval', interval=1000, n_intervals=0),
], fluid=True)


# Callbacks for dynamic updates
@app.callback(
    Output('current-emotion', 'children'),
    [Input('update-interval', 'n_intervals')]
)
def update_current_emotion(n):
    if not predicted_emotions:
        return "N/A"
    return predicted_emotions[-1]


@app.callback(
    Output('emotion-heatmap', 'figure'),
    [Input('update-interval', 'n_intervals')]
)
def update_heatmap(n):
    if not predicted_emotions:
        return go.Figure()

    emotion_counts = {label: predicted_emotions.count(label) for label in emotion_labels}
    heatmap_data = np.array([list(emotion_counts.values())])
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=emotion_labels,
        y=["Frequency"],
        colorscale='Blues',
        zmin=0,
        zmax=10,
        colorbar=dict(title="Frequency")
    ))
    fig.update_layout(
        title="Emotion Frequency Heatmap",
        xaxis=dict(tickangle=-45),
        font=dict(color='black'),
        plot_bgcolor='white',
        paper_bgcolor='white',
    )
    return fig


@app.callback(
    Output('emotion-line-chart', 'figure'),
    [Input('update-interval', 'n_intervals')]
)
def update_line_chart(n):
    timestamps.append(n)
    for label in emotion_labels:
        emotion_trends[label].append(predicted_emotions.count(label))
    fig = go.Figure()
    for label in emotion_labels:
        fig.add_trace(go.Scatter(x=list(timestamps), y=list(emotion_trends[label]), mode='lines', name=label))
    fig.update_layout(
        title="Emotion Trends Over Time",
        xaxis_title="Time Intervals",
        yaxis_title="Frequency",
        font=dict(color='black'),
        plot_bgcolor='white',
        paper_bgcolor='white',
    )
    return fig


@app.callback(
    Output('emotion-pie-chart', 'figure'),
    [Input('update-interval', 'n_intervals')]
)
def update_pie_chart(n):
    if not predicted_emotions:
        return go.Figure()

    emotion_counts = {label: predicted_emotions.count(label) for label in emotion_labels}
    fig = go.Figure(data=[go.Pie(labels=list(emotion_counts.keys()), values=list(emotion_counts.values()), hole=0.4)])
    fig.update_layout(
        title="Emotion Distribution",
        font=dict(color='black'),
        paper_bgcolor='white',
    )
    return fig


@app.callback(
    Output('video-feed', 'src'),
    [Input('update-interval', 'n_intervals')]
)
def update_video_feed(n):
    frame, emotion = process_frame()
    if emotion:
        predicted_emotions.append(emotion)
    return f"data:image/jpeg;base64,{encode_frame_to_base64(frame)}"


# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)