import dash
from dash import dcc, html
import dash_bootstrap_components as dbc

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define the layout of the landing page
app.layout = html.Div([
    # Header Section with Navigation
    html.Nav([
        dbc.NavbarSimple(
            children=[
                dbc.NavItem(dbc.NavLink("Home", href="#home")),
                dbc.NavItem(dbc.NavLink("About", href="#about")),
                dbc.NavItem(dbc.NavLink("How It Works", href="#how-it-works")),
                dbc.NavItem(dbc.NavLink("Demo", href="#demo")),
                dbc.NavItem(dbc.NavLink("Contact", href="#contact")),
            ],
            brand="FER System",
            brand_href="#",
            color="dark",
            dark=True,
        ),
    ]),

    # Hero Section (Above the Fold)
    html.Section([
        html.Div([
            html.H1("Recognize Emotions in Real-Time with Advanced AI"),
            html.P("Our Face Emotion Recognition system uses deep learning to detect and analyze emotions in real-time from facial expressions."),
            html.Button("Try the Demo", id="try-demo", n_clicks=0),
        ], className="text-center py-5", style={"background": "#e0e0e0"}),
    ]),

    # About Section
    html.Section([
        html.Div([
            html.H2("What is FER?"),
            html.P("A face emotion recognition system that detects emotions based on facial expressions using AI technology."),
        ], className="text-center py-5"),
    ]),

    # How It Works Section
    html.Section([
        html.Div([
            html.H2("How Does It Work?"),
            html.Ol([
                html.Li("Capture video from your webcam."),
                html.Li("The system detects faces."),
                html.Li("Emotions are detected from facial expressions."),
                html.Li("Results are shown instantly."),
            ]),
        ], className="text-center py-5"),
    ]),

    # Demo Section
    html.Section([
        html.Div([
            html.H2("Try It Yourself!"),
            html.P("Click below to start the demo and see how the system recognizes emotions."),
            html.Button("Start the Demo", id="start-demo", n_clicks=0),
        ], className="text-center py-5"),
    ]),

    # Footer Section
    html.Footer([
        html.Div([
            html.P("Contact us at: info@fer.com"),
            html.P("© 2025 Face Emotion Recognition. All rights reserved."),
        ], className="text-center py-3", style={"background": "#333", "color": "white"}),
    ], className="footer")
])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)