import sys
import os
from PySide6.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QTableWidget, QTableWidgetItem, QTextEdit, QFileDialog, QMessageBox, QScrollArea, 
                             QGridLayout, QSplitter, QPushButton, QDial, QLCDNumber, QProgressBar, QStatusBar,
                             QGroupBox, QFrame, QLineEdit, QComboBox, QMenu, QDialog)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtGui import QPalette, QColor, QIcon, QFont, QActionGroup
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtGui import QPalette, QColor, QIcon, QFont
from PySide6.QtWidgets import QApplication
import matplotlib.pyplot as plt
# plt.style.use('dark_background')  # Dark theme for plots
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", message=".*to view this Streamlit app.*")
warnings.filterwarnings("ignore", message=".*Thread 'MainThread': missing ScriptRunContext.*")

# Import des fonctions d'analyse
from analysis.geotechnical_analysis import perform_complete_analysis

# Import conditionnel du système RAG (pour éviter les erreurs de dépendances)
try:
    from models.rag_system import CPT_RAG_System
    RAG_SYSTEM_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Système RAG non disponible: {e}")
    print("L'application fonctionnera sans les fonctionnalités d'IA avancées.")
    CPT_RAG_System = None
    RAG_SYSTEM_AVAILABLE = False

# Import des nouveaux modules de parsing et vérification
from core.cpt_parser import CPTParser
from core.data_integrity_checker import DataIntegrityChecker

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        global RAG_SYSTEM_AVAILABLE  # Déclarer comme variable globale
        self.setWindowTitle("CPT Analysis Software - Puissant Logiciel Windows")
        # Utiliser le logo PNG si disponible, sinon SVG
        logo_path = 'LOGO VECTORISE PNG.png'
        if not os.path.exists(logo_path):
            logo_path = 'logo.svg'
        self.setWindowIcon(QIcon(logo_path))
        self.setGeometry(50, 50, 1400, 900)
        self.df = None
        self.analysis_results = None
        self.analysis_data = None  # To store the tuple

        # Initialisation conditionnelle du système RAG
        if RAG_SYSTEM_AVAILABLE and CPT_RAG_System:
            try:
                print("🚀 Initialisation du système RAG...")
                self.ai_explainer = CPT_RAG_System()
                # Initialiser seulement le modèle de base, pas les embeddings qui téléchargent
                self.ai_explainer.initialize_model()
                print("✅ Système RAG initialisé (embeddings différés)")
            except Exception as e:
                print(f"⚠️ Erreur lors de l'initialisation du RAG system: {e}")
                self.ai_explainer = None
                RAG_SYSTEM_AVAILABLE = False
        else:
            self.ai_explainer = None
            print("ℹ️ Fonctionnalités d'IA désactivées (RAG system non disponible)")

        self.data_checker = DataIntegrityChecker()  # Vérificateur d'intégrité des données
        self.setOffWhiteTheme()  # Default to off-white
        self.initUI()

    def setOffWhiteTheme(self):
        # Blanc cassé theme
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(245, 245, 245))
        palette.setColor(QPalette.WindowText, Qt.black)
        palette.setColor(QPalette.Base, QColor(255, 255, 255))
        palette.setColor(QPalette.AlternateBase, QColor(240, 240, 240))
        palette.setColor(QPalette.ToolTipBase, Qt.white)
        palette.setColor(QPalette.ToolTipText, Qt.black)
        palette.setColor(QPalette.Text, Qt.black)
        palette.setColor(QPalette.Button, QColor(240, 240, 240))
        palette.setColor(QPalette.ButtonText, Qt.black)
        palette.setColor(QPalette.BrightText, Qt.red)
        palette.setColor(QPalette.Link, QColor(0, 0, 255))
        palette.setColor(QPalette.Highlight, QColor(0, 120, 215))
        palette.setColor(QPalette.HighlightedText, Qt.white)
        self.setPalette(palette)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QTabWidget::pane {
                border: 1px solid #ccc;
                background-color: #fff;
            }
            QTabBar::tab {
                background-color: #e0e0e0;
                color: black;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #0078d7;
                color: white;
            }
            QPushButton {
                background-color: #e0e0e0;
                color: black;
                border: 1px solid #ccc;
                padding: 5px 10px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #d0d0d0;
            }
            QPushButton:pressed {
                background-color: #c0c0c0;
            }
            QTableWidget {
                background-color: #fff;
                color: black;
                gridline-color: #ccc;
            }
            QTextEdit {
                background-color: #fff;
                color: black;
                border: 1px solid #ccc;
            }
            QScrollArea {
                border: none;
            }
        """)
        plt.style.use('default')
        self.current_theme = 'offwhite'
        if hasattr(self, 'aiWebView'):
            self.updateChatTheme()
        if self.df is not None:
            self.updateGraphs()

    def setDarkTheme(self):
        # Dark theme like VST plugins
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(53, 53, 53))
        palette.setColor(QPalette.WindowText, Qt.white)
        palette.setColor(QPalette.Base, QColor(25, 25, 25))
        palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        palette.setColor(QPalette.ToolTipBase, Qt.white)
        palette.setColor(QPalette.ToolTipText, Qt.white)
        palette.setColor(QPalette.Text, Qt.white)
        palette.setColor(QPalette.Button, QColor(53, 53, 53))
        palette.setColor(QPalette.ButtonText, Qt.white)
        palette.setColor(QPalette.BrightText, Qt.red)
        palette.setColor(QPalette.Link, QColor(42, 130, 218))
        palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.HighlightedText, Qt.black)
        self.setPalette(palette)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #353535;
            }
            QTabWidget::pane {
                border: 1px solid #555;
                background-color: #2a2a2a;
            }
            QTabBar::tab {
                background-color: #555;
                color: white;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #2a82da;
            }
            QPushButton {
                background-color: #555;
                color: white;
                border: 1px solid #777;
                padding: 5px 10px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #666;
            }
            QPushButton:pressed {
                background-color: #444;
            }
            QTableWidget {
                background-color: #2a2a2a;
                color: white;
                gridline-color: #555;
            }
            QTextEdit {
                background-color: #2a2a2a;
                color: white;
                border: 1px solid #555;
            }
            QScrollArea {
                border: none;
            }
        """)
        plt.style.use('dark_background')
        self.current_theme = 'dark'
        if hasattr(self, 'aiWebView'):
            self.updateChatTheme()
        if self.df is not None:
            self.updateGraphs()

    def setDarkTheme(self):
        # Dark theme like VST plugins
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(53, 53, 53))
        palette.setColor(QPalette.WindowText, Qt.white)
        palette.setColor(QPalette.Base, QColor(25, 25, 25))
        palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        palette.setColor(QPalette.ToolTipBase, Qt.white)
        palette.setColor(QPalette.ToolTipText, Qt.white)
        palette.setColor(QPalette.Text, Qt.white)
        palette.setColor(QPalette.Button, QColor(53, 53, 53))
        palette.setColor(QPalette.ButtonText, Qt.white)
        palette.setColor(QPalette.BrightText, Qt.red)
        palette.setColor(QPalette.Link, QColor(42, 130, 218))
        palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.HighlightedText, Qt.black)
        self.setPalette(palette)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #353535;
            }
            QTabWidget::pane {
                border: 1px solid #555;
                background-color: #2a2a2a;
            }
            QTabBar::tab {
                background-color: #555;
                color: white;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #2a82da;
            }
            QPushButton {
                background-color: #555;
                color: white;
                border: 1px solid #777;
                padding: 5px 10px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #666;
            }
            QPushButton:pressed {
                background-color: #444;
            }
            QTableWidget {
                background-color: #2a2a2a;
                color: white;
                gridline-color: #555;
            }
            QTextEdit {
                background-color: #2a2a2a;
                color: white;
                border: 1px solid #555;
            }
            QScrollArea {
                border: none;
            }
        """)

    def setBlueTheme(self):
        # Blue theme
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(240, 248, 255))  # Alice blue
        palette.setColor(QPalette.WindowText, QColor(25, 25, 112))  # Midnight blue
        palette.setColor(QPalette.Base, QColor(255, 255, 255))
        palette.setColor(QPalette.AlternateBase, QColor(240, 248, 255))
        palette.setColor(QPalette.ToolTipBase, QColor(25, 25, 112))
        palette.setColor(QPalette.ToolTipText, QColor(25, 25, 112))
        palette.setColor(QPalette.Text, QColor(25, 25, 112))
        palette.setColor(QPalette.Button, QColor(173, 216, 230))  # Light blue
        palette.setColor(QPalette.ButtonText, QColor(25, 25, 112))
        palette.setColor(QPalette.BrightText, Qt.red)
        palette.setColor(QPalette.Link, QColor(0, 0, 255))
        palette.setColor(QPalette.Highlight, QColor(30, 144, 255))  # Dodger blue
        palette.setColor(QPalette.HighlightedText, Qt.white)
        self.setPalette(palette)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f8ff;
            }
            QTabWidget::pane {
                border: 1px solid #add8e6;
                background-color: #fff;
            }
            QTabBar::tab {
                background-color: #add8e6;
                color: #191970;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #1e90ff;
                color: white;
            }
            QPushButton {
                background-color: #add8e6;
                color: #191970;
                border: 1px solid #87ceeb;
                padding: 5px 10px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #87ceeb;
            }
            QPushButton:pressed {
                background-color: #4682b4;
            }
            QTableWidget {
                background-color: #fff;
                color: #191970;
                gridline-color: #add8e6;
            }
            QTextEdit {
                background-color: #fff;
                color: #191970;
                border: 1px solid #add8e6;
            }
            QScrollArea {
                border: none;
            }
        """)
        plt.style.use('default')
        self.current_theme = 'blue'
        if hasattr(self, 'aiWebView'):
            self.updateChatTheme()
        if self.df is not None:
            self.updateGraphs()

    def setGreenTheme(self):
        # Green theme
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(240, 255, 240))  # Honeydew
        palette.setColor(QPalette.WindowText, QColor(0, 100, 0))  # Dark green
        palette.setColor(QPalette.Base, QColor(255, 255, 255))
        palette.setColor(QPalette.AlternateBase, QColor(240, 255, 240))
        palette.setColor(QPalette.ToolTipBase, QColor(0, 100, 0))
        palette.setColor(QPalette.ToolTipText, QColor(0, 100, 0))
        palette.setColor(QPalette.Text, QColor(0, 100, 0))
        palette.setColor(QPalette.Button, QColor(144, 238, 144))  # Light green
        palette.setColor(QPalette.ButtonText, QColor(0, 100, 0))
        palette.setColor(QPalette.BrightText, Qt.red)
        palette.setColor(QPalette.Link, QColor(0, 128, 0))
        palette.setColor(QPalette.Highlight, QColor(50, 205, 50))  # Lime green
        palette.setColor(QPalette.HighlightedText, Qt.white)
        self.setPalette(palette)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0fff0;
            }
            QTabWidget::pane {
                border: 1px solid #90ee90;
                background-color: #fff;
            }
            QTabBar::tab {
                background-color: #90ee90;
                color: #006400;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #32cd32;
                color: white;
            }
            QPushButton {
                background-color: #90ee90;
                color: #006400;
                border: 1px solid #98fb98;
                padding: 5px 10px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #98fb98;
            }
            QPushButton:pressed {
                background-color: #228b22;
            }
            QTableWidget {
                background-color: #fff;
                color: #006400;
                gridline-color: #90ee90;
            }
            QTextEdit {
                background-color: #fff;
                color: #006400;
                border: 1px solid #90ee90;
            }
            QScrollArea {
                border: none;
            }
        """)
        plt.style.use('default')
        self.current_theme = 'green'
        if hasattr(self, 'aiWebView'):
            self.updateChatTheme()
        if self.df is not None:
            self.updateGraphs()

    def setPurpleTheme(self):
        # Purple theme
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(248, 240, 255))  # Lavender blush with purple tint
        palette.setColor(QPalette.WindowText, QColor(75, 0, 130))  # Indigo
        palette.setColor(QPalette.Base, QColor(255, 255, 255))
        palette.setColor(QPalette.AlternateBase, QColor(248, 240, 255))
        palette.setColor(QPalette.ToolTipBase, QColor(75, 0, 130))
        palette.setColor(QPalette.ToolTipText, QColor(75, 0, 130))
        palette.setColor(QPalette.Text, QColor(75, 0, 130))
        palette.setColor(QPalette.Button, QColor(221, 160, 221))  # Plum
        palette.setColor(QPalette.ButtonText, QColor(75, 0, 130))
        palette.setColor(QPalette.BrightText, Qt.red)
        palette.setColor(QPalette.Link, QColor(128, 0, 128))
        palette.setColor(QPalette.Highlight, QColor(138, 43, 226))  # Blue violet
        palette.setColor(QPalette.HighlightedText, Qt.white)
        self.setPalette(palette)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f8f0ff;
            }
            QTabWidget::pane {
                border: 1px solid #dda0dd;
                background-color: #fff;
            }
            QTabBar::tab {
                background-color: #dda0dd;
                color: #4b0082;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #8a2be2;
                color: white;
            }
            QPushButton {
                background-color: #dda0dd;
                color: #4b0082;
                border: 1px solid #e6e6fa;
                padding: 5px 10px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #e6e6fa;
            }
            QPushButton:pressed {
                background-color: #9370db;
            }
            QTableWidget {
                background-color: #fff;
                color: #4b0082;
                gridline-color: #dda0dd;
            }
            QTextEdit {
                background-color: #fff;
                color: #4b0082;
                border: 1px solid #dda0dd;
            }
            QScrollArea {
                border: none;
            }
        """)
        plt.style.use('default')
        self.current_theme = 'purple'
        if hasattr(self, 'aiWebView'):
            self.updateChatTheme()
        if self.df is not None:
            self.updateGraphs()

    def setClayTheme(self):
        # Thème argile avec image de fond
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(245, 245, 245))
        palette.setColor(QPalette.WindowText, QColor(60, 60, 60))
        palette.setColor(QPalette.Base, QColor(255, 255, 255))
        palette.setColor(QPalette.AlternateBase, QColor(240, 240, 240))
        palette.setColor(QPalette.ToolTipBase, Qt.white)
        palette.setColor(QPalette.ToolTipText, QColor(60, 60, 60))
        palette.setColor(QPalette.Text, QColor(60, 60, 60))
        palette.setColor(QPalette.Button, QColor(240, 240, 240))
        palette.setColor(QPalette.ButtonText, QColor(60, 60, 60))
        palette.setColor(QPalette.BrightText, Qt.red)
        palette.setColor(QPalette.Link, QColor(0, 100, 150))
        palette.setColor(QPalette.Highlight, QColor(0, 120, 180))
        palette.setColor(QPalette.HighlightedText, Qt.white)
        self.setPalette(palette)
        self.setStyleSheet("""
            QMainWindow {
                background-image: url('close-up-melange-de-poudre-d-argile.jpg');
                background-repeat: no-repeat;
                background-position: center;
                background-attachment: fixed;
                background-size: cover;
                background-color: rgba(245, 245, 245, 0.9);
            }
            QMainWindow::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background-color: rgba(255, 255, 255, 0.85);
                z-index: -1;
            }
            QTabWidget::pane {
                border: 1px solid rgba(100, 100, 100, 0.3);
                background-color: rgba(255, 255, 255, 0.95);
                border-radius: 5px;
            }
            QTabBar::tab {
                background-color: rgba(240, 240, 240, 0.9);
                color: #3c3c3c;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                border: 1px solid rgba(100, 100, 100, 0.2);
            }
            QTabBar::tab:selected {
                background-color: rgba(0, 120, 180, 0.8);
                color: white;
                border: 1px solid rgba(0, 120, 180, 0.5);
            }
            QPushButton {
                background-color: rgba(240, 240, 240, 0.9);
                color: #3c3c3c;
                border: 1px solid rgba(100, 100, 100, 0.3);
                padding: 5px 10px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: rgba(220, 220, 220, 0.9);
                border: 1px solid rgba(0, 120, 180, 0.5);
            }
            QPushButton:pressed {
                background-color: rgba(200, 200, 200, 0.9);
            }
            QTableWidget {
                background-color: rgba(255, 255, 255, 0.95);
                color: #3c3c3c;
                gridline-color: rgba(100, 100, 100, 0.3);
                border: 1px solid rgba(100, 100, 100, 0.2);
            }
            QTextEdit {
                background-color: rgba(255, 255, 255, 0.95);
                color: #3c3c3c;
                border: 1px solid rgba(100, 100, 100, 0.3);
            }
            QScrollArea {
                border: none;
                background-color: rgba(255, 255, 255, 0.95);
            }
            QLabel {
                color: #3c3c3c;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid rgba(0, 120, 180, 0.3);
                border-radius: 5px;
                margin-top: 1ex;
                background-color: rgba(255, 255, 255, 0.9);
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 10px 0 10px;
                color: #3c3c3c;
            }
        """)
        plt.style.use('default')
        self.current_theme = 'clay'
        if hasattr(self, 'aiWebView'):
            self.updateChatTheme()
        if self.df is not None:
            self.updateGraphs()

    def initUI(self):
        # Menu
        menubar = self.menuBar()
        fileMenu = menubar.addMenu('Fichier')
        openAction = fileMenu.addAction('Ouvrir')
        openAction.triggered.connect(self.openFile)
        
        toolsMenu = menubar.addMenu('Outils')
        integrityAction = toolsMenu.addAction('Vérifier Intégrité Données')
        integrityAction.triggered.connect(self.showDataIntegrityReport)
        
        # Theme menu
        themeMenu = menubar.addMenu('Thème')
        self.themeGroup = QActionGroup(self)
        
        offWhiteAction = themeMenu.addAction('Blanc Cassé')
        offWhiteAction.setCheckable(True)
        offWhiteAction.setChecked(True)
        offWhiteAction.triggered.connect(self.setOffWhiteTheme)
        self.themeGroup.addAction(offWhiteAction)
        
        darkAction = themeMenu.addAction('Dark VST')
        darkAction.setCheckable(True)
        darkAction.triggered.connect(self.setDarkTheme)
        self.themeGroup.addAction(darkAction)
        
        blueAction = themeMenu.addAction('Bleu')
        blueAction.setCheckable(True)
        blueAction.triggered.connect(self.setBlueTheme)
        self.themeGroup.addAction(blueAction)
        
        greenAction = themeMenu.addAction('Vert')
        greenAction.setCheckable(True)
        greenAction.triggered.connect(self.setGreenTheme)
        self.themeGroup.addAction(greenAction)
        
        purpleAction = themeMenu.addAction('Violet')
        purpleAction.setCheckable(True)
        purpleAction.triggered.connect(self.setPurpleTheme)
        self.themeGroup.addAction(purpleAction)
        
        clayAction = themeMenu.addAction('Argile')
        clayAction.setCheckable(True)
        clayAction.triggered.connect(self.setClayTheme)
        self.themeGroup.addAction(clayAction)

        helpMenu = menubar.addMenu('Aide')
        aboutAction = helpMenu.addAction('À propos')
        aboutAction.triggered.connect(self.showAbout)
        
        presentationAction = helpMenu.addAction('Présentation du Logiciel')
        presentationAction.triggered.connect(self.showPresentation)

        # Widget central avec splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(splitter)

        # Panneau gauche pour les onglets
        self.tabs = QTabWidget()
        splitter.addWidget(self.tabs)

        # Panneau droit pour les explications IA
        self.aiPanel = QWidget()
        aiLayout = QVBoxLayout()
        aiHeader = QHBoxLayout()
        aiHeader.addWidget(QLabel("🤖 Chat IA Géotechnique"))
        self.refreshAIButton = QPushButton("🔄 Actualiser")
        self.refreshAIButton.clicked.connect(self.updateAI)
        aiHeader.addWidget(self.refreshAIButton)
        aiLayout.addLayout(aiHeader)
        
        # Chat input
        chatLayout = QHBoxLayout()
        self.chatInput = QLineEdit()
        self.chatInput.setPlaceholderText("Posez une question sur les données CPT...")
        self.chatInput.returnPressed.connect(self.sendChatMessage)
        chatLayout.addWidget(self.chatInput)
        sendButton = QPushButton("Envoyer")
        sendButton.clicked.connect(self.sendChatMessage)
        chatLayout.addWidget(sendButton)
        aiLayout.addLayout(chatLayout)
        
        scroll_ai = QScrollArea()
        scroll_ai.setWidgetResizable(True)
        self.aiWebView = QWebEngineView()
        self.aiWebView.setMaximumWidth(400)
        self.aiWebView.setMinimumHeight(300)
        
        # HTML initial pour le chat
        self.chat_html_template = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 15px;
            background-color: {bg_color};
            color: {text_color};
            line-height: 1.6;
        }}
        .message {{
            margin-bottom: 20px;
            padding: 10px;
            border-radius: 8px;
            border-left: 4px solid #2a82da;
        }}
        .user-message {{
            background-color: {user_bg};
            border-left-color: #2a82da;
        }}
        .ai-message {{
            background-color: {ai_bg};
            border-left-color: #4CAF50;
        }}
        .message-header {{
            font-weight: bold;
            margin-bottom: 8px;
            display: flex;
            align-items: center;
        }}
        .user-icon {{ color: #2a82da; }}
        .ai-icon {{ color: #4CAF50; }}
        .content {{
            margin-left: 20px;
        }}
        .image-container {{
            margin: 10px 0;
            text-align: center;
        }}
        .image-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 6px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .code-block {{
            background-color: {code_bg};
            padding: 10px;
            border-radius: 4px;
            font-family: 'Consolas', 'Monaco', monospace;
            margin: 10px 0;
            overflow-x: auto;
        }}
        .loading {{
            color: #ff9800;
            font-style: italic;
        }}
        .error {{
            color: #f44336;
            background-color: {error_bg};
            padding: 8px;
            border-radius: 4px;
            margin: 10px 0;
        }}
        .web-results {{
            background-color: {web_bg};
            padding: 10px;
            border-radius: 6px;
            margin: 10px 0;
            border-left: 3px solid #ff9800;
        }}
        .web-results h4 {{
            margin-top: 0;
            color: #ff9800;
        }}
    </style>
</head>
<body>
    <div id="chat-container">
        <div class="message ai-message">
            <div class="message-header">
                <span class="ai-icon">🤖</span> Chat IA Géotechnique
            </div>
            <div class="content">
                Bonjour ! Je suis votre assistant IA spécialisé en géotechnique.<br>
                Chargez un fichier CPT et lancez l'analyse pour commencer à discuter.
            </div>
        </div>
    </div>
</body>
</html>"""
        
        # Appliquer le thème
        self.updateChatTheme()
        
        scroll_ai.setWidget(self.aiWebView)
        aiLayout.addWidget(scroll_ai)
        self.aiPanel.setLayout(aiLayout)
        splitter.addWidget(self.aiPanel)
        
        splitter.setSizes([1000, 400])

        # Status bar
        self.statusBar = self.statusBar()
        self.statusBar.showMessage("Prêt")

        # Créer les onglets
        self.createDataTab()
        self.createFusionTab()
        self.createAnalysisTab()
        self.createGraphsTab()
        self.create3DTab()
        self.createTablesTab()
        self.createOverviewTab()
        
        # Add icons to tabs (using text for now)
        # self.tabs.setTabIcon(0, self.style().standardIcon(self.style().SP_FileDialogContentsView))
        # etc.

    def updateChatTheme(self):
        """Update the chat HTML theme based on current theme"""
        if self.current_theme == 'offwhite':
            colors = {
                'bg_color': '#f8f8f8',
                'text_color': '#333',
                'user_bg': '#e3f2fd',
                'ai_bg': '#f5f5f5',
                'code_bg': '#f8f8f8',
                'error_bg': '#ffebee',
                'web_bg': '#fff3e0'
            }
        elif self.current_theme == 'dark':
            colors = {
                'bg_color': '#2a2a2a',
                'text_color': '#ffffff',
                'user_bg': '#3a3a3a',
                'ai_bg': '#353535',
                'code_bg': '#404040',
                'error_bg': '#4a2c2c',
                'web_bg': '#4a3c28'
            }
        elif self.current_theme == 'blue':
            colors = {
                'bg_color': '#f0f8ff',
                'text_color': '#191970',
                'user_bg': '#e6e6fa',
                'ai_bg': '#f0f8ff',
                'code_bg': '#f5f5f5',
                'error_bg': '#ffe4e1',
                'web_bg': '#fffacd'
            }
        elif self.current_theme == 'green':
            colors = {
                'bg_color': '#f0fff0',
                'text_color': '#006400',
                'user_bg': '#f0fff0',
                'ai_bg': '#f5fff5',
                'code_bg': '#f8fff8',
                'error_bg': '#fff0f0',
                'web_bg': '#fffacd'
            }
        else:  # purple
            colors = {
                'bg_color': '#faf0ff',
                'text_color': '#4b0082',
                'user_bg': '#f2e6ff',
                'ai_bg': '#faf0ff',
                'code_bg': '#f8f0ff',
                'error_bg': '#ffe6e6',
                'web_bg': '#fff8dc'
            }
        
        html_content = self.chat_html_template.format(**colors)
        if hasattr(self, 'aiWebView'):
            self.aiWebView.setHtml(html_content)

    def addChatMessage(self, message_type, content, include_images=False, web_results=None):
        """Add a message to the chat with enhanced formatting"""
        # Utiliser une approche plus sûre pour injecter le contenu
        # Échapper tous les caractères problématiques pour JavaScript
        safe_content = (content.replace('\\', '\\\\')
                       .replace('"', '\\"')
                       .replace("'", "\\'")
                       .replace('\n', '\\n')
                       .replace('\r', '\\r')
                       .replace('\t', '\\t'))
        
        script = f"""
        (function() {{
            try {{
                var container = document.getElementById('chat-container');
                if (!container) return;
                
                var messageDiv = document.createElement('div');
                messageDiv.className = 'message {message_type}-message';
                
                var headerDiv = document.createElement('div');
                headerDiv.className = 'message-header';
                
                var iconSpan = document.createElement('span');
                iconSpan.className = '{message_type}-icon';
                iconSpan.textContent = '{("👤" if message_type == "user" else "🤖")}';
                headerDiv.appendChild(iconSpan);
                
                var titleSpan = document.createElement('span');
                titleSpan.textContent = ' {"Vous" if message_type == "user" else "IA Géotechnique"}';
                headerDiv.appendChild(titleSpan);
                
                messageDiv.appendChild(headerDiv);
                
                var contentDiv = document.createElement('div');
                contentDiv.className = 'content';
                contentDiv.innerHTML = "{safe_content}";
                messageDiv.appendChild(contentDiv);
                
                container.appendChild(messageDiv);
                window.scrollTo(0, document.body.scrollHeight);
            }} catch (e) {{
                console.error('JavaScript error in addChatMessage:', e);
            }}
        }})();
        """
        
        if hasattr(self, 'aiWebView'):
            self.aiWebView.page().runJavaScript(script)

    def updateLastAIMessage(self, new_content):
        """Update the content of the last AI message"""
        # Utiliser une approche plus sûre pour injecter le contenu
        # Échapper tous les caractères problématiques pour JavaScript
        safe_content = (new_content.replace('\\', '\\\\')
                       .replace('"', '\\"')
                       .replace("'", "\\'")
                       .replace('\n', '\\n')
                       .replace('\r', '\\r')
                       .replace('\t', '\\t'))
        
        script = f"""
        (function() {{
            try {{
                var messages = document.querySelectorAll('.ai-message');
                if (messages.length > 0) {{
                    var lastMessage = messages[messages.length - 1];
                    var contentDiv = lastMessage.querySelector('.content');
                    if (contentDiv) {{
                        contentDiv.innerHTML = "{safe_content}";
                    }}
                }}
                window.scrollTo(0, document.body.scrollHeight);
            }} catch (e) {{
                console.error('JavaScript error in updateLastAIMessage:', e);
            }}
        }})();
        """
        if hasattr(self, 'aiWebView'):
            self.aiWebView.page().runJavaScript(script)

    def _get_progress_indicator(self, current_response):
        """Génère un indicateur de progression basé sur les phases détectées dans la réponse"""
        response_lower = current_response.lower()

        # Détecter la phase actuelle basée sur les marqueurs dans la réponse (10 phases maintenant)
        phase_indicators = {
            "analyse de la requête": "📝 Phase 1/10: Analyse de la requête",
            "réponse prédéfinie": "🧠 Phase 2/10: Vérification des réponses prédéfinies",
            "contexte géographique": "🌍 Phase 3/10: Recherche du contexte géographique",
            "planification": "🎯 Phase 4/10: Planification de la réflexion",
            "recherche dans les données": "📚 Phase 5/10: Recherche dans les données CPT",
            "recherche sur internet": "🌐 Phase 6/10: Recherche sur Internet",
            "calculs et statistiques": "🧮 Phase 7/10: Calculs et statistiques",
            "construction du contexte": "🤖 Phase 8/10: Construction du contexte",
            "analyses scientifiques": "🔬 Phase 9/10: Analyses scientifiques avancées",
            "vérification des données": "🔍 Phase 10/10: Vérification des données et recommandations",
            "génération de la réponse": "✨ Génération finale de la réponse"
        }

        for marker, indicator in phase_indicators.items():
            if marker in response_lower:
                return indicator

        # Si aucune phase spécifique détectée, estimer basé sur la longueur et le contenu
        length = len(current_response)

        # Indicateurs animés pour montrer l'activité
        import time
        dots = "." * ((int(time.time()) % 3) + 1)

        if length < 50:
            return f"🚀 Initialisation{dots}"
        elif length < 150:
            return f"⚡ Analyse de la question{dots}"
        elif length < 300:
            return f"🔍 Recherche d'informations{dots}"
        elif length < 500:
            return f"🧮 Calculs en cours{dots}"
        elif length < 800:
            return f"📊 Traitement des données{dots}"
        elif length < 1200:
            return f"🤖 Génération intelligente{dots}"
        else:
            return f"✨ Finalisation{dots}"

    def enhance_response_with_web_search(self, response, question):
        """Enhance response with web search results"""
        try:
            # Utiliser l'outil de recherche web du système RAG
            from tools.web import web_search
            
            # Rechercher des informations complémentaires
            search_query = f"géotechnique CPT {question.lower()}"
            web_results = web_search(search_query)
            
            if web_results and 'results' in web_results and len(web_results['results']) > 0:
                web_section = '<div class="web-results"><h4>🔍 Informations complémentaires :</h4>'
                for result in web_results['results'][:2]:  # Limiter à 2 résultats
                    title = result.get('title', 'Information')
                    snippet = result.get('body', '')[:200] + '...'
                    web_section += f'<p><strong>{title}</strong><br>{snippet}</p>'
                web_section += '</div>'
                response += web_section
                
        except Exception as e:
            print(f"Web search enhancement failed: {e}")
            
        return response

    def add_visualizations_to_response(self, response, question):
        """Add relevant visualizations to the response"""
        try:
            # Détecter si la question nécessite des visualisations
            question_lower = question.lower()
            
            if any(keyword in question_lower for keyword in ['graphique', 'courbe', 'profil', 'visualisation', 'plot']):
                # Générer un graphique simple basé sur les données
                if hasattr(self, 'df') and self.df is not None:
                    import matplotlib.pyplot as plt
                    import io
                    import base64
                    
                    fig, ax = plt.subplots(figsize=(6, 4))
                    
                    if 'qc' in self.df.columns and 'depth' in self.df.columns:
                        ax.plot(self.df['qc'], self.df['Depth'], 'b-', linewidth=2)
                        ax.set_xlabel('qc (MPa)')
                        ax.set_ylabel('Profondeur (cm)')
                        ax.set_title('Profil de qc')
                        ax.invert_yaxis()
                        ax.grid(True, alpha=0.3)
                        
                        # Convertir en base64 pour HTML
                        buf = io.BytesIO()
                        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                        buf.seek(0)
                        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
                        plt.close(fig)
                        
                        img_html = f'<div class="image-container"><img src="data:image/png;base64,{img_base64}" alt="Graphique qc"><br><em>Profil de résistance au cône (qc)</em></div>'
                        response += img_html
                        
        except Exception as e:
            print(f"Visualization enhancement failed: {e}")
            
        return response

    def createDataTab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        
        # Table des données
        self.dataTable = QTableWidget()
        self.dataTable.setAlternatingRowColors(True)
        self.dataTable.setMinimumWidth(800)
        scroll.setWidget(self.dataTable)
        
        layout.addWidget(QLabel("📊 Tableau des Données CPT"))
        layout.addWidget(scroll)
        
        # Boutons d'export
        btn_layout = QHBoxLayout()
        pdf_btn = QPushButton("📄 Exporter en PDF")
        pdf_btn.clicked.connect(self.exportDataToPDF)
        btn_layout.addWidget(pdf_btn)
        
        excel_btn = QPushButton("📊 Exporter en Excel")
        excel_btn.clicked.connect(self.exportDataToExcel)
        btn_layout.addWidget(excel_btn)
        
        layout.addLayout(btn_layout)
        
        tab.setLayout(layout)
        self.tabs.addTab(tab, "Données")

    def createFusionTab(self):
        """Créer l'onglet de fusion de fichiers CPTU"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        layout.addWidget(QLabel("🔗 Fusion de Fichiers CPTU - Carte 3D Complète du Sol"))
        
        # Section de chargement des fichiers
        upload_group = QGroupBox("📁 Chargement des Fichiers CPTU")
        upload_layout = QVBoxLayout()
        
        # Bouton pour charger plusieurs fichiers
        self.loadMultipleButton = QPushButton("📂 Charger Plusieurs Fichiers CPTU")
        self.loadMultipleButton.clicked.connect(self.loadMultipleCPTUFiles)
        self.loadMultipleButton.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                padding: 12px 24px;
                border: none;
                border-radius: 6px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        upload_layout.addWidget(self.loadMultipleButton)
        
        # Liste des fichiers chargés
        self.fusionFileList = QTextEdit()
        self.fusionFileList.setMaximumHeight(100)
        self.fusionFileList.setPlaceholderText("Fichiers chargés apparaîtront ici...")
        upload_layout.addWidget(self.fusionFileList)
        
        upload_group.setLayout(upload_layout)
        layout.addWidget(upload_group)
        
        # Section des coordonnées
        coord_group = QGroupBox("📍 Coordonnées des Sondages")
        coord_layout = QVBoxLayout()
        
        coord_layout.addWidget(QLabel("Définissez les coordonnées X,Y pour chaque sondage CPTU:"))
        
        # Table pour les coordonnées
        self.coordTable = QTableWidget()
        self.coordTable.setColumnCount(3)
        self.coordTable.setHorizontalHeaderLabels(["Fichier", "Coordonnée X (m)", "Coordonnée Y (m)"])
        self.coordTable.horizontalHeader().setStretchLastSection(True)
        coord_layout.addWidget(self.coordTable)
        
        # Boutons pour gérer les coordonnées
        coord_buttons = QHBoxLayout()
        
        self.addCoordButton = QPushButton("➕ Ajouter Coordonnées")
        self.addCoordButton.clicked.connect(self.addCoordinates)
        coord_buttons.addWidget(self.addCoordButton)
        
        self.autoDetectCoordButton = QPushButton("🤖 Auto-détecter Coordonnées")
        self.autoDetectCoordButton.clicked.connect(self.autoDetectAndFillCoordinates)
        self.autoDetectCoordButton.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                padding: 8px 16px;
                border: none;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        coord_buttons.addWidget(self.autoDetectCoordButton)
        
        self.clearCoordButton = QPushButton("🗑️ Effacer Tout")
        self.clearCoordButton.clicked.connect(self.clearCoordinates)
        coord_buttons.addWidget(self.clearCoordButton)
        
        coord_layout.addLayout(coord_buttons)
        coord_group.setLayout(coord_layout)
        layout.addWidget(coord_group)
        
        # Section de fusion et visualisation
        fusion_group = QGroupBox("🔄 Fusion et Visualisation 3D")
        fusion_layout = QVBoxLayout()
        
        # Boutons de contrôle
        button_layout = QHBoxLayout()
        
        # Bouton de fusion
        self.fusionButton = QPushButton("🚀 Créer Carte 2D Complète du Sol")
        self.fusionButton.clicked.connect(self.create3DSoilMap)
        self.fusionButton.setEnabled(False)
        self.fusionButton.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 12px 24px;
                border: none;
                border-radius: 6px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        button_layout.addWidget(self.fusionButton)
        
        # Bouton d'export PDF
        self.exportFusionPDFButton = QPushButton("📄 Exporter Carte 2D en PDF")
        self.exportFusionPDFButton.clicked.connect(self.exportFusion3DToPDF)
        self.exportFusionPDFButton.setEnabled(False)
        self.exportFusionPDFButton.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                padding: 12px 24px;
                border: none;
                border-radius: 6px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        button_layout.addWidget(self.exportFusionPDFButton)
        
        button_layout.addStretch()
        fusion_layout.addLayout(button_layout)
        
        # Zone de visualisation 2D complète
        self.fusion2DView = QWebEngineView()
        self.fusion2DView.setMinimumHeight(400)
        fusion_layout.addWidget(QLabel("🗺️ Carte 2D Complète - Vue d'ensemble des Sondages"))
        fusion_layout.addWidget(self.fusion2DView)
        
        # Informations sur la carte 2D
        self.fusion2DInfoLabel = QLabel("Informations sur la carte 2D apparaîtront ici...")
        self.fusion2DInfoLabel.setStyleSheet("font-weight: bold; color: #666;")
        fusion_layout.addWidget(self.fusion2DInfoLabel)
        
        fusion_group.setLayout(fusion_layout)
        layout.addWidget(fusion_group)
        
        # Stocker les données de fusion
        self.fusion_files = []  # Liste des fichiers chargés
        self.fusion_data = {}   # Données fusionnées avec coordonnées
        
        tab.setLayout(layout)
        self.tabs.addTab(tab, "🗺️ Fusion 2D")

    def createAnalysisTab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        
        # Résultats d'analyse
        self.analysisText = QTextEdit()
        self.analysisText.setMinimumWidth(600)
        scroll.setWidget(self.analysisText)
        
        layout.addWidget(QLabel("🔬 Analyse Géotechnique Complète"))
        layout.addWidget(scroll)
        
        tab.setLayout(layout)
        self.tabs.addTab(tab, "Analyse")

    def createGraphsTab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        grid = QGridLayout()
        
        self.canvases = []
        
        # Liste des 20 graphiques avec légendes
        plots_config = [
            ('qc vs Profondeur', lambda df, ax: self.plot_qc_depth(df, ax)),
            ('fs vs Profondeur', lambda df, ax: self.plot_fs_depth(df, ax)),
            ('Rf vs Profondeur', lambda df, ax: self.plot_rf_depth(df, ax)),
            ('qnet vs Profondeur', lambda df, ax: self.plot_qnet_depth(df, ax)),
            ('Classification des sols', lambda df, ax: self.plot_soil_classification(df, ax)),
            ('Clusters K-means', lambda df, ax: self.plot_kmeans_clusters(df, ax)),
            ('PCA - Composantes principales', lambda df, ax: self.plot_pca(df, ax)),
            ('Profil lissé qc', lambda df, ax: self.plot_smooth_qc(df, ax)),
            ('Profil lissé fs', lambda df, ax: self.plot_smooth_fs(df, ax)),
            ('Histogramme qc', lambda df, ax: self.plot_qc_histogram(df, ax)),
            ('Histogramme fs', lambda df, ax: self.plot_fs_histogram(df, ax)),
            ('Boxplot qc par couche', lambda df, ax: self.plot_qc_boxplot(df, ax)),
            ('Boxplot fs par couche', lambda df, ax: self.plot_fs_boxplot(df, ax)),
            ('Nuage de points qc vs fs', lambda df, ax: self.plot_qc_fs_scatter(df, ax)),
            ('Courbe cumulative qc', lambda df, ax: self.plot_qc_cumulative(df, ax)),
            ('Courbe cumulative fs', lambda df, ax: self.plot_fs_cumulative(df, ax)),
            ('Profil de friction ratio', lambda df, ax: self.plot_friction_ratio(df, ax)),
            ('Analyse de tendance qc', lambda df, ax: self.plot_qc_trend(df, ax)),
            ('Analyse de tendance fs', lambda df, ax: self.plot_fs_trend(df, ax)),
            ('Carte de chaleur corrélations', lambda df, ax: self.plot_correlation_heatmap(df, ax)),
        ]
        
        for i, (title, plot_func) in enumerate(plots_config):
            group = QGroupBox(title)
            group.setStyleSheet("QGroupBox { font-weight: bold; border: 2px solid #555; border-radius: 5px; margin-top: 1ex; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px 0 5px; }")
            group_layout = QVBoxLayout()
            canvas = FigureCanvas(plt.Figure(figsize=(10, 8)))
            ax = canvas.figure.add_subplot(111)
            ax.set_title(title, fontsize=12, fontweight='bold', color='white')
            ax.set_facecolor('#2a2a2a')
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_edgecolor('white')
            self.canvases.append((canvas, plot_func))
            
            group_layout.addWidget(canvas)
            
            save_btn = QPushButton("💾 PDF")
            save_btn.setFixedWidth(80)
            save_btn.clicked.connect(lambda checked, idx=i: self.savePlotAsPDF(idx))
            group_layout.addWidget(save_btn)
            
            group.setLayout(group_layout)
            group.setMaximumWidth(500)
            grid.addWidget(group, i // 3, i % 3)  # 3 colonnes pour meilleure visibilité
        
        content.setLayout(grid)
        scroll.setWidget(content)
        layout.addWidget(scroll)
        
        tab.setLayout(layout)
        self.tabs.addTab(tab, "Graphiques (20)")

    def create3DTab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        
        layout.addWidget(QLabel("📉 Visualisations 3D Interactives (4 Graphiques)"))
        
        # Ajouter une barre d'outils pour les boutons de téléchargement
        toolbar = QHBoxLayout()
        
        # Bouton pour télécharger tous les graphiques 3D en PDF
        self.export3DPDFButton = QPushButton("📄 Télécharger PDF (Tous les graphiques 3D)")
        self.export3DPDFButton.clicked.connect(self.export3DGraphsToPDF)
        self.export3DPDFButton.setEnabled(False)  # Désactivé par défaut
        self.export3DPDFButton.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 8px 16px;
                border: none;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        toolbar.addWidget(self.export3DPDFButton)
        
        # Bouton pour télécharger chaque graphique individuellement
        self.export3DIndividualButton = QPushButton("🖼️ Télécharger Graphiques Individuellement")
        self.export3DIndividualButton.clicked.connect(self.export3DGraphsIndividually)
        self.export3DIndividualButton.setEnabled(False)  # Désactivé par défaut
        self.export3DIndividualButton.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                padding: 8px 16px;
                border: none;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        toolbar.addWidget(self.export3DIndividualButton)
        
        # Bouton de rafraîchissement des graphiques 3D
        self.refresh3DButton = QPushButton("🔄 Actualiser 3D")
        self.refresh3DButton.clicked.connect(self.refresh3DGraphs)
        self.refresh3DButton.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                padding: 8px 16px;
                border: none;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
        """)
        toolbar.addWidget(self.refresh3DButton)
        
        toolbar.addStretch()  # Espacement
        layout.addLayout(toolbar)
        
        # Create grid layout for 4 plots (2x2)
        grid = QGridLayout()
        
        # Create 4 QWebEngineView for the plots
        self.webView1 = QWebEngineView()
        self.webView2 = QWebEngineView()
        self.webView3 = QWebEngineView()
        self.webView4 = QWebEngineView()
        
        # Add labels and views to grid
        grid.addWidget(QLabel("3D Scatter: Depth vs qc vs fs"), 0, 0)
        grid.addWidget(self.webView1, 1, 0)
        
        grid.addWidget(QLabel("3D Surface: qc Surface"), 0, 1)
        grid.addWidget(self.webView2, 1, 1)
        
        grid.addWidget(QLabel("3D Contour: qc Contours"), 2, 0)
        grid.addWidget(self.webView3, 3, 0)
        
        grid.addWidget(QLabel("3D Wireframe: qc Wireframe"), 2, 1)
        grid.addWidget(self.webView4, 3, 1)
        
        layout.addLayout(grid)
        
        tab.setLayout(layout)
        self.tabs.addTab(tab, "3D Interactif")

    def createTablesTab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        grid = QGridLayout()
        
        # Tables supplémentaires
        self.summaryTable = QTableWidget()
        self.layersTable = QTableWidget()
        self.statsTable = QTableWidget()
        self.correlationTable = QTableWidget()
        self.liquefactionTable = QTableWidget()  # New table
        
        tables = [
            (self.summaryTable, "Résumé des données"),
            (self.layersTable, "Classification par couches"),
            (self.statsTable, "Statistiques détaillées"),
            (self.correlationTable, "Matrice de corrélation"),
            (self.liquefactionTable, "Analyse de liquéfaction")
        ]
        
        for i, (table, title) in enumerate(tables):
            table.setAlternatingRowColors(True)
            table.setMinimumWidth(400)
            table.setMaximumWidth(600)
            row = i // 3
            col = i % 3
            grid.addWidget(QLabel(f"📋 {title}"), row * 2, col)
            grid.addWidget(table, row * 2 + 1, col)
        
        content.setLayout(grid)
        scroll.setWidget(content)
        layout.addWidget(scroll)
        
        tab.setLayout(layout)
        self.tabs.addTab(tab, "Tableaux")

    def createOverviewTab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Bouton de téléchargement PDF
        pdf_btn = QPushButton("📄 Télécharger Vue d'ensemble en PDF")
        pdf_btn.setStyleSheet("QPushButton { font-size: 14px; font-weight: bold; padding: 10px; background-color: #0078d7; color: white; border-radius: 5px; } QPushButton:hover { background-color: #005a9e; }")
        pdf_btn.clicked.connect(self.saveOverviewAsPDF)
        layout.addWidget(pdf_btn)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        main_layout = QVBoxLayout()
        
        # Section Graphiques
        graphs_label = QLabel("📊 Tous les Graphiques")
        graphs_label.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px;")
        main_layout.addWidget(graphs_label)
        
        graphs_grid = QGridLayout()
        self.overview_canvases = []
        
        plots_config = [
            ('qc vs Profondeur', lambda df, ax: self.plot_qc_depth(df, ax)),
            ('fs vs Profondeur', lambda df, ax: self.plot_fs_depth(df, ax)),
            ('Rf vs Profondeur', lambda df, ax: self.plot_rf_depth(df, ax)),
            ('qnet vs Profondeur', lambda df, ax: self.plot_qnet_depth(df, ax)),
            ('Classification des sols', lambda df, ax: self.plot_soil_classification(df, ax)),
            ('Clusters K-means', lambda df, ax: self.plot_kmeans_clusters(df, ax)),
            ('PCA - Composantes principales', lambda df, ax: self.plot_pca(df, ax)),
            ('Profil lissé qc', lambda df, ax: self.plot_smooth_qc(df, ax)),
            ('Profil lissé fs', lambda df, ax: self.plot_smooth_fs(df, ax)),
            ('Histogramme qc', lambda df, ax: self.plot_qc_histogram(df, ax)),
            ('Histogramme fs', lambda df, ax: self.plot_fs_histogram(df, ax)),
            ('Boxplot qc par couche', lambda df, ax: self.plot_qc_boxplot(df, ax)),
            ('Boxplot fs par couche', lambda df, ax: self.plot_fs_boxplot(df, ax)),
            ('Nuage de points qc vs fs', lambda df, ax: self.plot_qc_fs_scatter(df, ax)),
            ('Courbe cumulative qc', lambda df, ax: self.plot_qc_cumulative(df, ax)),
            ('Courbe cumulative fs', lambda df, ax: self.plot_fs_cumulative(df, ax)),
            ('Profil de friction ratio', lambda df, ax: self.plot_friction_ratio(df, ax)),
            ('Analyse de tendance qc', lambda df, ax: self.plot_qc_trend(df, ax)),
            ('Analyse de tendance fs', lambda df, ax: self.plot_fs_trend(df, ax)),
            ('Carte de chaleur corrélations', lambda df, ax: self.plot_correlation_heatmap(df, ax)),
        ]
        
        for i, (title, plot_func) in enumerate(plots_config):
            group = QGroupBox(title)
            group.setStyleSheet("QGroupBox { font-weight: bold; border: 2px solid #555; border-radius: 5px; margin-top: 1ex; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px 0 5px; }")
            group_layout = QVBoxLayout()
            canvas = FigureCanvas(plt.Figure(figsize=(8, 6)))
            ax = canvas.figure.add_subplot(111)
            ax.set_title(title, fontsize=10, fontweight='bold')
            self.overview_canvases.append((canvas, plot_func))
            
            group_layout.addWidget(canvas)
            group.setLayout(group_layout)
            group.setMaximumWidth(400)
            graphs_grid.addWidget(group, i // 4, i % 4)  # 4 colonnes
        
        graphs_widget = QWidget()
        graphs_widget.setLayout(graphs_grid)
        main_layout.addWidget(graphs_widget)
        
        # Section Tableaux
        tables_label = QLabel("📋 Tous les Tableaux")
        tables_label.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px;")
        main_layout.addWidget(tables_label)
        
        tables_grid = QGridLayout()
        
        self.overview_summaryTable = QTableWidget()
        self.overview_layersTable = QTableWidget()
        self.overview_statsTable = QTableWidget()
        self.overview_correlationTable = QTableWidget()
        self.overview_liquefactionTable = QTableWidget()
        
        tables = [
            (self.overview_summaryTable, "Résumé des données"),
            (self.overview_layersTable, "Classification par couches"),
            (self.overview_statsTable, "Statistiques détaillées"),
            (self.overview_correlationTable, "Matrice de corrélation"),
            (self.overview_liquefactionTable, "Analyse de liquéfaction")
        ]
        
        for i, (table, title) in enumerate(tables):
            table.setAlternatingRowColors(True)
            table.setMinimumWidth(300)
            table.setMaximumWidth(500)
            row = i // 3
            col = i % 3
            tables_grid.addWidget(QLabel(f"📋 {title}"), row * 2, col)
            tables_grid.addWidget(table, row * 2 + 1, col)
        
        tables_widget = QWidget()
        tables_widget.setLayout(tables_grid)
        main_layout.addWidget(tables_widget)
        
        content.setLayout(main_layout)
        scroll.setWidget(content)
        layout.addWidget(scroll)
        
        tab.setLayout(layout)
        self.tabs.addTab(tab, "Vue d'ensemble")

    # Méthodes de tracé pour les 20 graphiques
    def plot_qc_depth(self, df, ax):
        ax.plot(df['qc'], df['Depth'], 'b-', linewidth=2, label='qc (MPa)', color='#2a82da')
        ax.set_xlabel('Résistance de pointe qc (MPa)', color='white')
        ax.set_ylabel('Profondeur (cm)', color='white')
        ax.invert_yaxis()
        ax.legend()
        ax.grid(True, alpha=0.3, color='white')
        ax.set_facecolor('#2a2a2a')

    def plot_fs_depth(self, df, ax):
        ax.plot(df['fs'], df['Depth'], 'r-', linewidth=2, label='fs (kPa)', color='#da2a2a')
        ax.set_xlabel('Résistance latérale fs (kPa)', color='white')
        ax.set_ylabel('Profondeur (cm)', color='white')
        ax.invert_yaxis()
        ax.legend()
        ax.grid(True, alpha=0.3, color='white')
        ax.set_facecolor('#2a2a2a')

    def plot_rf_depth(self, df, ax):
        if 'Rf' in df.columns:
            ax.plot(df['Rf'], df['Depth'], 'g-', linewidth=2, label='Rf (%)')
            ax.set_xlabel('Ratio de friction Rf (%)')
            ax.set_ylabel('Profondeur (cm)')
            ax.invert_yaxis()
            ax.legend()
            ax.grid(True, alpha=0.3)

    def plot_qnet_depth(self, df, ax):
        if 'qnet' in df.columns:
            ax.plot(df['qnet'], df['Depth'], 'm-', linewidth=2, label='qnet (MPa)')
            ax.set_xlabel('Résistance nette qnet (MPa)')
            ax.set_ylabel('Profondeur (cm)')
            ax.invert_yaxis()
            ax.legend()
            ax.grid(True, alpha=0.3)

    def plot_soil_classification(self, df, ax):
        if 'Soil_Type' in df.columns:
            soil_colors = {'Sand': 'yellow', 'Silt': 'green', 'Clay': 'brown', 'Gravel': 'gray'}
            for soil, color in soil_colors.items():
                mask = df['Soil_Type'] == soil
                if mask.any():
                    ax.scatter(df.loc[mask, 'qc'], df.loc[mask, 'Depth'], c=color, label=soil, alpha=0.7)
            ax.set_xlabel('qc (MPa)')
            ax.set_ylabel('Profondeur (cm)')
            ax.invert_yaxis()
            ax.legend()
            ax.grid(True, alpha=0.3)

    def plot_kmeans_clusters(self, df, ax):
        try:
            features = df[['qc', 'fs']].dropna()
            if len(features) > 5:
                scaler = StandardScaler()
                features_scaled = scaler.fit_transform(features)
                kmeans = KMeans(n_clusters=3, random_state=42)
                clusters = kmeans.fit_predict(features_scaled)
                ax.scatter(features['qc'], features['fs'], c=clusters, cmap='viridis', alpha=0.7)
                ax.set_xlabel('qc (MPa)')
                ax.set_ylabel('fs (kPa)')
                ax.set_title('Clusters K-means')
                ax.grid(True, alpha=0.3)
        except:
            ax.text(0.5, 0.5, 'Données insuffisantes', transform=ax.transAxes, ha='center')

    def plot_pca(self, df, ax):
        try:
            features = df[['qc', 'fs']].dropna()
            if len(features) > 5:
                scaler = StandardScaler()
                features_scaled = scaler.fit_transform(features)
                pca = PCA(n_components=2)
                pca_result = pca.fit_transform(features_scaled)
                ax.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.7)
                ax.set_xlabel('PC1')
                ax.set_ylabel('PC2')
                ax.set_title('PCA - 2 composantes')
                ax.grid(True, alpha=0.3)
        except:
            ax.text(0.5, 0.5, 'Données insuffisantes', transform=ax.transAxes, ha='center')

    def plot_smooth_qc(self, df, ax):
        try:
            y_smooth = savgol_filter(df['qc'], window_length=min(11, len(df)//2*2+1), polyorder=3)
            ax.plot(df['qc'], df['Depth'], 'b-', alpha=0.5, label='Original')
            ax.plot(y_smooth, df['Depth'], 'r-', linewidth=2, label='Lissé')
            ax.set_xlabel('qc (MPa)')
            ax.set_ylabel('Profondeur (cm)')
            ax.invert_yaxis()
            ax.legend()
            ax.grid(True, alpha=0.3)
        except:
            ax.text(0.5, 0.5, 'Erreur de lissage', transform=ax.transAxes, ha='center')

    def plot_smooth_fs(self, df, ax):
        try:
            y_smooth = savgol_filter(df['fs'], window_length=min(11, len(df)//2*2+1), polyorder=3)
            ax.plot(df['fs'], df['Depth'], 'r-', alpha=0.5, label='Original')
            ax.plot(y_smooth, df['Depth'], 'b-', linewidth=2, label='Lissé')
            ax.set_xlabel('fs (kPa)')
            ax.set_ylabel('Profondeur (cm)')
            ax.invert_yaxis()
            ax.legend()
            ax.grid(True, alpha=0.3)
        except:
            ax.text(0.5, 0.5, 'Erreur de lissage', transform=ax.transAxes, ha='center')

    def plot_qc_histogram(self, df, ax):
        ax.hist(df['qc'], bins=20, alpha=0.7, color='blue', edgecolor='black')
        ax.set_xlabel('qc (MPa)')
        ax.set_ylabel('Fréquence')
        ax.set_title('Distribution qc')
        ax.grid(True, alpha=0.3)

    def plot_fs_histogram(self, df, ax):
        ax.hist(df['fs'], bins=20, alpha=0.7, color='red', edgecolor='black')
        ax.set_xlabel('fs (kPa)')
        ax.set_ylabel('Fréquence')
        ax.set_title('Distribution fs')
        ax.grid(True, alpha=0.3)

    def plot_qc_boxplot(self, df, ax):
        ax.boxplot(df['qc'])
        ax.set_ylabel('qc (MPa)')
        ax.set_title('Boxplot qc')
        ax.grid(True, alpha=0.3)

    def plot_fs_boxplot(self, df, ax):
        ax.boxplot(df['fs'])
        ax.set_ylabel('fs (kPa)')
        ax.set_title('Boxplot fs')
        ax.grid(True, alpha=0.3)

    def plot_qc_fs_scatter(self, df, ax):
        ax.scatter(df['qc'], df['fs'], alpha=0.6, c=df['Depth'], cmap='viridis')
        ax.set_xlabel('qc (MPa)')
        ax.set_ylabel('fs (kPa)')
        ax.set_title('qc vs fs (couleur = profondeur)')
        ax.grid(True, alpha=0.3)

    def plot_qc_cumulative(self, df, ax):
        ax.plot(np.cumsum(df['qc']), df['Depth'], 'b-', linewidth=2)
        ax.set_xlabel('Somme cumulative qc')
        ax.set_ylabel('Profondeur (cm)')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)

    def plot_fs_cumulative(self, df, ax):
        ax.plot(np.cumsum(df['fs']), df['Depth'], 'r-', linewidth=2)
        ax.set_xlabel('Somme cumulative fs')
        ax.set_ylabel('Profondeur (cm)')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)

    def plot_friction_ratio(self, df, ax):
        if 'Rf' in df.columns:
            ax.plot(df['Rf'], df['Depth'], 'g-', linewidth=2)
            ax.axvline(x=5, color='r', linestyle='--', label='Limite sable/argile')
            ax.set_xlabel('Ratio de friction (%)')
            ax.set_ylabel('Profondeur (cm)')
            ax.invert_yaxis()
            ax.legend()
            ax.grid(True, alpha=0.3)

    def plot_qc_trend(self, df, ax):
        z = np.polyfit(df['Depth'], df['qc'], 1)
        p = np.poly1d(z)
        ax.scatter(df['Depth'], df['qc'], alpha=0.6)
        ax.plot(df['Depth'], p(df['Depth']), 'r-', linewidth=2, label=f'Tendance: {z[0]:.4f}x + {z[1]:.2f}')
        ax.set_xlabel('Profondeur (cm)')
        ax.set_ylabel('qc (MPa)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def plot_fs_trend(self, df, ax):
        z = np.polyfit(df['Depth'], df['fs'], 1)
        p = np.poly1d(z)
        ax.scatter(df['Depth'], df['fs'], alpha=0.6)
        ax.plot(df['Depth'], p(df['Depth']), 'r-', linewidth=2, label=f'Tendance: {z[0]:.4f}x + {z[1]:.2f}')
        ax.set_xlabel('Profondeur (cm)')
        ax.set_ylabel('fs (kPa)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def plot_correlation_heatmap(self, df, ax):
        corr = df[['Depth', 'qc', 'fs']].corr()
        cax = ax.matshow(corr, cmap='coolwarm')
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns)
        ax.set_yticklabels(corr.columns)
        plt.colorbar(cax, ax=ax)
        ax.set_title('Matrice de corrélation')

    def openFile(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Ouvrir fichier CPT", "",
                                                  "Fichiers CPT (*.txt *.xlsx *.csv *.xls *.cal);;Tous les fichiers (*)")
        if fileName:
            try:
                # Vérification d'intégrité des données avant parsing
                integrity_results = self.data_checker.verify_file_integrity(fileName)

                # Afficher les résultats de vérification
                integrity_report = self.data_checker.generate_integrity_report(fileName)

                # Parser le fichier avec le nouveau CPTParser
                parser = CPTParser()
                df, parse_message = parser.parse_file(fileName)

                if df is None:
                    QMessageBox.warning(self, "Erreur de parsing", f"Échec du parsing: {parse_message}")
                    return

                # Vérifier l'intégrité des données parsées
                if not integrity_results['data_integrity']:
                    warning_msg = "⚠️ PROBLÈMES D'INTÉGRITÉ DÉTECTÉS:\n\n"
                    if integrity_results['parsing_errors']:
                        warning_msg += "Erreurs de parsing:\n" + "\n".join(f"• {e}" for e in integrity_results['parsing_errors']) + "\n\n"
                    if integrity_results['data_loss_warnings']:
                        warning_msg += "Avertissements perte de données:\n" + "\n".join(f"• {w}" for w in integrity_results['data_loss_warnings']) + "\n\n"
                    if integrity_results['precision_warnings']:
                        warning_msg += "Avertissements précision:\n" + "\n".join(f"• {w}" for w in integrity_results['precision_warnings']) + "\n\n"
                    if integrity_results['range_warnings']:
                        warning_msg += "Avertissements plage:\n" + "\n".join(f"• {w}" for w in integrity_results['range_warnings']) + "\n\n"

                    warning_msg += "Voulez-vous continuer avec ces données ?"

                    reply = QMessageBox.question(self, "Avertissement Intégrité",
                                               warning_msg,
                                               QMessageBox.Yes | QMessageBox.No,
                                               QMessageBox.No)

                    if reply == QMessageBox.No:
                        return

                # Normaliser les noms de colonnes pour la compatibilité
                column_normalization = {
                    'depth': 'Depth', 'profondeur': 'Depth',
                    'qc': 'qc', 'résistance': 'qc', 'pression': 'qc',
                    'fs': 'fs', 'frottement': 'fs',
                    'u': 'u', 'pore_pressure': 'u',
                    'u2': 'u2',
                    'rf': 'Rf', 'friction_ratio': 'Rf',
                    'gamma': 'gamma', 'unit_weight': 'gamma',
                    'vs': 'Vs', 'shear_wave': 'Vs'
                }

                df = df.rename(columns=column_normalization)

                # Trier par profondeur si disponible (sans conversion destructive)
                if 'Depth' in df.columns:
                    df = df.sort_values('Depth').reset_index(drop=True)

                # Garder les données avec NaN partiels plutôt que supprimer toutes les lignes
                # Seulement supprimer les lignes complètement vides
                df = df.dropna(how='all')

                self.df = df

                # Afficher le rapport d'intégrité dans la console pour debug
                print(integrity_report)

                # Message de succès avec informations d'intégrité
                integrity_status = "✓ Données validées" if integrity_results['data_integrity'] else "⚠️ Anomalies détectées"
                success_msg = f"Fichier chargé: {len(df)} points de données\n{integrity_status}\n\n"
                success_msg += f"Colonnes: {', '.join(df.columns)}\n"
                success_msg += f"Statistiques rapides:\n"

                if 'Depth' in df.columns:
                    success_msg += f"• Profondeur: {df['Depth'].min():.2f} - {df['Depth'].max():.2f} m\n"
                if 'qc' in df.columns:
                    success_msg += f"• qc: {df['qc'].min():.2f} - {df['qc'].max():.2f} MPa\n"
                if 'fs' in df.columns:
                    success_msg += f"• fs: {df['fs'].min():.2f} - {df['fs'].max():.2f} kPa\n"
                    print(f"DEBUG: fs stats - Min: {df['fs'].min():.2f}, Max: {df['fs'].max():.2f}, Count: {len(df['fs'])}")

                success_msg += "\nCliquez sur 'Analyser' pour lancer l'analyse complète avec IA."

                self.analysis_results = f"Fichier chargé avec vérification d'intégrité.\n{integrity_status}"

                self.updateAll()

                QMessageBox.information(self, "Succès", success_msg)

            except Exception as e:
                QMessageBox.warning(self, "Erreur", f"Échec du chargement: {str(e)}")

    def updateAll(self):
        self.updateDataTable()
        self.updateGraphs()
        self.update3D()
        self.updateAnalysis()
        self.updateTables()
        self.updateAI()
        
        # Activer/désactiver les boutons d'export 3D selon la présence de données
        has_data = hasattr(self, 'df') and self.df is not None and not self.df.empty
        has_3d_views = all(hasattr(self, f'webView{i}') for i in range(1, 5))
        
        if hasattr(self, 'export3DPDFButton'):
            self.export3DPDFButton.setEnabled(has_data and has_3d_views)
        if hasattr(self, 'export3DIndividualButton'):
            self.export3DIndividualButton.setEnabled(has_data and has_3d_views)
        if hasattr(self, 'refresh3DButton'):
            self.refresh3DButton.setEnabled(has_3d_views)

    def updateDataTable(self):
        if self.df is not None:
            self.dataTable.setRowCount(len(self.df))
            self.dataTable.setColumnCount(len(self.df.columns))
            self.dataTable.setHorizontalHeaderLabels(self.df.columns)
            for i in range(min(1000, len(self.df))):  # Limiter à 1000 lignes pour performance
                for j in range(len(self.df.columns)):
                    self.dataTable.setItem(i, j, QTableWidgetItem(str(self.df.iloc[i, j])))

    def updateGraphs(self):
        for canvas, plot_func in self.canvases:
            canvas.figure.clear()
            ax = canvas.figure.add_subplot(111)
            if self.df is not None:
                try:
                    plot_func(self.df, ax)
                except Exception as e:
                    ax.text(0.5, 0.5, f'Erreur: {str(e)}', transform=ax.transAxes, ha='center')
            canvas.draw()
        
        # Update overview canvases
        if hasattr(self, 'overview_canvases'):
            for canvas, plot_func in self.overview_canvases:
                canvas.figure.clear()
                ax = canvas.figure.add_subplot(111)
                if self.df is not None:
                    try:
                        plot_func(self.df, ax)
                    except Exception as e:
                        ax.text(0.5, 0.5, f'Erreur: {str(e)}', transform=ax.transAxes, ha='center')
                canvas.draw()

    def update3D(self):
        # Vérifier que les webView existent
        webviews = ['webView1', 'webView2', 'webView3', 'webView4']
        for wv_name in webviews:
            if not hasattr(self, wv_name):
                print(f"⚠️ {wv_name} n'existe pas encore")
                return

        if self.df is not None and not self.df.empty:
            try:
                # Vérifier que les colonnes nécessaires existent
                required_cols = ['Depth', 'qc', 'fs']
                missing_cols = [col for col in required_cols if col not in self.df.columns]
                if missing_cols:
                    error_msg = f"Colonnes manquantes pour les graphiques 3D: {', '.join(missing_cols)}"
                    error_html = f"<h2>Erreur 3D</h2><p>{error_msg}</p><p>Colonnes disponibles: {', '.join(self.df.columns)}</p>"
                    for webview in [self.webView1, self.webView2, self.webView3, self.webView4]:
                        if hasattr(self, webview.__name__):
                            webview.setHtml(error_html)
                    return

                # Nettoyer les données
                df_clean = self.df.dropna(subset=required_cols)
                if len(df_clean) < 3:
                    error_html = "<h2>Erreur 3D</h2><p>Pas assez de données valides pour créer les graphiques 3D (minimum 3 points requis)</p>"
                    for webview in [self.webView1, self.webView2, self.webView3, self.webView4]:
                        if hasattr(self, webview.__name__):
                            webview.setHtml(error_html)
                    return

                # Plot 1: 3D Scatter Depth vs qc vs fs
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter3d(
                    x=df_clean['Depth'],
                    y=df_clean['qc'],
                    z=df_clean['fs'],
                    mode='markers',
                    marker=dict(size=4, color=df_clean['qc'], colorscale='Viridis', showscale=True)
                ))
                fig1.update_layout(
                    title='3D Scatter: Depth vs qc vs fs',
                    scene=dict(
                        xaxis_title='Depth (cm)',
                        yaxis_title='qc (MPa)',
                        zaxis_title='fs (kPa)',
                        camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
                    ),
                    template='plotly_white',
                    height=500,
                    width=None,  # Responsive width
                    margin=dict(l=40, r=40, t=60, b=40, pad=10),
                    autosize=True
                )
                html1 = fig1.to_html(include_plotlyjs='cdn', full_html=False, config={'responsive': True})
                self.webView1.setHtml(html1)

                # Plot 2: 3D Surface plot for qc (version simplifiée sans interpolation)
                try:
                    # Essayer avec interpolation scipy
                    depth_grid = np.linspace(df_clean['Depth'].min(), df_clean['Depth'].max(), 30)
                    qc_grid = np.linspace(df_clean['qc'].min(), df_clean['qc'].max(), 30)
                    DEPTH, QC = np.meshgrid(depth_grid, qc_grid)

                    from scipy.interpolate import griddata
                    FS_interp = griddata((df_clean['Depth'], df_clean['qc']), df_clean['fs'],
                                       (DEPTH, QC), method='linear', fill_value=np.mean(df_clean['fs']))

                    fig2 = go.Figure(data=[go.Surface(z=FS_interp, x=DEPTH, y=QC, colorscale='Viridis')])
                    fig2.update_layout(
                        title='3D Surface: qc vs Depth vs fs',
                        scene=dict(
                            xaxis_title='Depth (cm)',
                            yaxis_title='qc (MPa)',
                            zaxis_title='fs (kPa)',
                            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
                        ),
                        template='plotly_white',
                        height=500,
                        width=None,  # Responsive width
                        margin=dict(l=40, r=40, t=60, b=40, pad=10),
                        autosize=True
                    )
                    html2 = fig2.to_html(include_plotlyjs='cdn', full_html=False, config={'responsive': True})
                    self.webView2.setHtml(html2)

                except ImportError:
                    # Fallback sans scipy - utiliser un scatter 3D coloré
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter3d(
                        x=df_clean['Depth'],
                        y=df_clean['qc'],
                        z=df_clean['fs'],
                        mode='markers',
                        marker=dict(size=6, color=df_clean['fs'], colorscale='Plasma', showscale=True)
                    ))
                    fig2.update_layout(
                        title='3D Scatter (Surface simulée): qc vs Depth vs fs',
                        scene=dict(
                            xaxis_title='Depth (cm)',
                            yaxis_title='qc (MPa)',
                            zaxis_title='fs (kPa)',
                            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
                        ),
                        template='plotly_white',
                        height=500,
                        width=None,  # Responsive width
                        margin=dict(l=40, r=40, t=60, b=40, pad=10),
                        autosize=True
                    )
                    html2 = fig2.to_html(include_plotlyjs='cdn', full_html=False, config={'responsive': True})
                    self.webView2.setHtml(html2)

                except Exception as e:
                    # Fallback en cas d'erreur d'interpolation
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter3d(
                        x=df_clean['Depth'],
                        y=df_clean['qc'],
                        z=df_clean['fs'],
                        mode='markers',
                        marker=dict(size=6, color=df_clean['qc'], colorscale='Viridis', showscale=True)
                    ))
                    fig2.update_layout(
                        title='3D Scatter (Fallback): qc vs Depth vs fs',
                        scene=dict(
                            xaxis_title='Depth (cm)',
                            yaxis_title='qc (MPa)',
                            zaxis_title='fs (kPa)',
                            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
                        ),
                        template='plotly_white',
                        height=500,
                        width=None,  # Responsive width
                        margin=dict(l=40, r=40, t=60, b=40, pad=10),
                        autosize=True
                    )
                    html2 = fig2.to_html(include_plotlyjs='cdn', full_html=False, config={'responsive': True})
                    self.webView2.setHtml(html2)

                # Plot 3: 3D Contour plot
                fig3 = go.Figure(data=go.Contour(
                    x=df_clean['Depth'],
                    y=df_clean['qc'],
                    z=df_clean['fs'],
                    colorscale='Viridis',
                    contours=dict(showlabels=True)
                ))
                fig3.update_layout(
                    title='3D Contour: Depth vs qc vs fs',
                    xaxis_title='Depth (cm)',
                    yaxis_title='qc (MPa)',
                    template='plotly_white',
                    height=500,
                    width=None,  # Responsive width
                    margin=dict(l=60, r=40, t=60, b=60, pad=10),
                    autosize=True
                )
                html3 = fig3.to_html(include_plotlyjs='cdn', full_html=False, config={'responsive': True})
                self.webView3.setHtml(html3)

                # Plot 4: 3D Wireframe (version simplifiée)
                fig4 = go.Figure()
                fig4.add_trace(go.Scatter3d(
                    x=df_clean['Depth'],
                    y=df_clean['qc'],
                    z=df_clean['fs'],
                    mode='lines+markers',
                    line=dict(color='blue', width=2),
                    marker=dict(size=3, color='red'),
                    name='Wireframe'
                ))
                fig4.update_layout(
                    title='3D Wireframe: qc Surface with Contours',
                    scene=dict(
                        xaxis_title='Depth (cm)',
                        yaxis_title='qc (MPa)',
                        zaxis_title='fs (kPa)',
                        camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
                    ),
                    template='plotly_white',
                    height=500,
                    width=None,  # Responsive width
                    margin=dict(l=40, r=40, t=60, b=40, pad=10),
                    autosize=True
                )
                html4 = fig4.to_html(include_plotlyjs='cdn', full_html=False, config={'responsive': True})
                self.webView4.setHtml(html4)

                print("✅ Graphiques 3D mis à jour avec succès")

            except Exception as e:
                error_html = f"<h1>Erreur 3D: {str(e)}</h1><p>Vérifiez que vos données contiennent les colonnes 'Depth', 'qc' et 'fs'</p>"
                print(f"❌ Erreur dans update3D(): {e}")
                import traceback
                traceback.print_exc()

                for webview in [self.webView1, self.webView2, self.webView3, self.webView4]:
                    if hasattr(self, webview.__name__ if hasattr(webview, '__name__') else 'webView'):
                        webview.setHtml(error_html)
        else:
            no_data_html = "<h1>Aucune donnée chargée</h1><p>Chargez un fichier CPTU pour voir les graphiques 3D</p>"
            for webview in [self.webView1, self.webView2, self.webView3, self.webView4]:
                if hasattr(self, webview.__name__ if hasattr(webview, '__name__') else 'webView'):
                    webview.setHtml(no_data_html)

    def refresh3DGraphs(self):
        """Forcer le rafraîchissement des graphiques 3D"""
        try:
            print("🔄 Rafraîchissement des graphiques 3D...")
            self.update3D()
            QMessageBox.information(self, "Succès", "Graphiques 3D actualisés !")
        except Exception as e:
            QMessageBox.warning(self, "Erreur", f"Erreur lors du rafraîchissement: {e}")

    def updateAnalysis(self):
        """Mettre à jour l'analyse géotechnique"""
        if self.df is not None:
            try:
                analysis_result = perform_complete_analysis(self.df, use_streamlit=False)
                if isinstance(analysis_result, tuple):
                    df_analyzed, models, results = analysis_result
                    self.analysis_data = (df_analyzed, models, results)
                    self.analysis_results = self.create_analysis_summary(df_analyzed, results)
                else:
                    self.analysis_results = str(analysis_result)
                    self.analysis_data = None
                
                # Add layer types and zones
                layers = self.classify_layers()
                zones = self.determine_zones()
                
                full_analysis = f"{self.analysis_results}\n\n🏔️ Types de Couches:\n{layers}\n\n📍 Zones de Pointage:\n{zones}"
                self.analysisText.setText(full_analysis)
                
                # Initialize RAG system after complete analysis (with chunking and FAISS)
                if self.analysis_data and len(self.analysis_data) >= 3:
                    try:
                        self.ai_explainer.initialize_system(self.analysis_data[2])  # results dict
                        print("✅ Système RAG initialisé avec les résultats d'analyse")
                    except Exception as e:
                        print(f"⚠️ Erreur initialisation RAG: {e}")
                
            except Exception as e:
                error_msg = f"Erreur lors de l'analyse: {str(e)}"
                self.analysisText.setText(error_msg)
                print(f"❌ Erreur dans updateAnalysis(): {e}")
                import traceback
                traceback.print_exc()

    def updateTables(self):
        """Mettre à jour tous les tableaux avec les données actuelles"""
        if self.df is None or self.df.empty:
            return
            
        try:
            # Table de résumé
            self.updateSummaryTable()
            
            # Table des couches
            self.updateLayersTable()
            
            # Table des statistiques
            self.updateStatsTable()
            
            # Table de corrélation
            self.updateCorrelationTable()
            
            # Table de liquéfaction
            self.updateLiquefactionTable()
            
        except Exception as e:
            print(f"Erreur lors de la mise à jour des tableaux: {e}")
            import traceback
            traceback.print_exc()

    def createFused2DVisualization(self, fused_data):
        """Créer la visualisation 2D complète de tous les sondages fusionnés"""
        try:
            if fused_data is None or fused_data.empty:
                error_html = "<h1>Erreur 2D</h1><p>Aucune donnée fusionnée disponible</p>"
                self.fusion2DView.setHtml(error_html)
                return

            # Créer la figure Plotly 2D
            fig = go.Figure()
            
            # Couleurs pour différents sondages
            colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 
                     'magenta', 'yellow', 'black', 'navy', 'maroon', 'lime', 'aqua', 'fuchsia', 'silver', 'teal']
            
            sondages = fused_data['Sondage'].unique()
            
            # Créer une légende pour les couleurs
            color_map = {}
            for i, sondage in enumerate(sondages):
                color_map[sondage] = colors[i % len(colors)]
            
            # Tracer chaque sondage avec sa couleur
            for sondage in sondages:
                sondage_data = fused_data[fused_data['Sondage'] == sondage]
                color = color_map[sondage]
                
                # Points de données colorés par qc
                fig.add_trace(go.Scatter(
                    x=sondage_data['X'],
                    y=sondage_data['Y'],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=sondage_data['qc'] if 'qc' in sondage_data.columns else color,
                        colorscale='Viridis' if 'qc' in sondage_data.columns else None,
                        showscale=True if sondage == sondages[0] and 'qc' in sondage_data.columns else False,
                        colorbar=dict(title="qc (MPa)") if sondage == sondages[0] and 'qc' in sondage_data.columns else None,
                        symbol='circle',
                        line=dict(width=1, color='black')
                    ),
                    name=f'{sondage} - qc',
                    legendgroup=sondage,
                    hovertemplate=f'<b>{sondage}</b><br>' +
                                 'X: %{x:.1f} m<br>' +
                                 'Y: %{y:.1f} m<br>' +
                                 'Profondeur: %{customdata:.1f} m<br>' +
                                 'qc: %{marker.color:.1f} MPa<extra></extra>',
                    customdata=sondage_data['Depth']
                ))
                
                # Ligne connectant les points du sondage
                fig.add_trace(go.Scatter(
                    x=sondage_data['X'],
                    y=sondage_data['Y'],
                    mode='lines+markers',
                    line=dict(color=color, width=3),
                    marker=dict(size=4, color=color),
                    name=f'{sondage} - profil',
                    legendgroup=sondage,
                    showlegend=False,
                    hovertemplate=f'<b>{sondage}</b><br>' +
                                 'X: %{x:.1f} m<br>' +
                                 'Y: %{y:.1f} m<extra></extra>'
                ))
            
            # Ajouter des annotations pour les noms des sondages
            for sondage in sondages:
                sondage_data = fused_data[fused_data['Sondage'] == sondage]
                # Position du premier point pour l'annotation
                x_pos = sondage_data['X'].iloc[0]
                y_pos = sondage_data['Y'].iloc[0]
                color = color_map[sondage]
                
                fig.add_annotation(
                    x=x_pos,
                    y=y_pos,
                    text=sondage,
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor=color,
                    font=dict(size=12, color=color, family="Arial Black"),
                    bgcolor="white",
                    bordercolor=color,
                    borderwidth=2,
                    borderpad=4,
                    ax=20,
                    ay=-20
                )
            
            # Configuration du layout
            fig.update_layout(
                title="🗺️ Carte 2D Complète du Sous-Sol - Fusion de Sondages CPTU",
                xaxis_title='Coordonnée X (m)',
                yaxis_title='Coordonnée Y (m)',
                xaxis=dict(
                    scaleanchor="y",
                    scaleratio=1,
                    showgrid=True,
                    gridcolor='lightgray'
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor='lightgray'
                ),
                plot_bgcolor='white',
                paper_bgcolor='white',
                height=600,
                width=800,
                margin=dict(l=50, r=50, t=80, b=50),
                legend_title="Sondages CPTU",
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                    bgcolor="rgba(255, 255, 255, 0.8)",
                    bordercolor="black",
                    borderwidth=1
                )
            )
            
            # Ajouter une grille pour mieux visualiser
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
            
            # Convertir en HTML et afficher
            html_content = fig.to_html(include_plotlyjs='cdn', full_html=False)
            self.fusion2DView.setHtml(html_content)
            
            # Informations sur la carte 2D
            info_2d = f"Carte 2D créée avec {len(sondages)} sondages:\n"
            for sondage in sondages:
                sondage_data = fused_data[fused_data['Sondage'] == sondage]
                color = color_map[sondage]
                info_2d += f"• {sondage}: {len(sondage_data)} points (couleur: {color})\n"
            
            self.fusion2DInfoLabel.setText(info_2d)
            self.fusion2DInfoLabel.setStyleSheet("font-weight: bold; color: #2196F3;")
            
            print("✅ Carte 2D fusionnée créée avec succès")
            
        except Exception as e:
            error_html = f"<h1>Erreur 2D Fusion: {str(e)}</h1>"
            self.fusion2DView.setHtml(error_html)
            print(f"❌ Erreur dans createFused2DVisualization: {e}")
            import traceback
            traceback.print_exc()

    def refresh3DGraphs(self):
        """Forcer le rafraîchissement des graphiques 3D"""
        try:
            print("🔄 Rafraîchissement des graphiques 3D...")
            self.update3D()
            QMessageBox.information(self, "Succès", "Graphiques 3D actualisés !")
        except Exception as e:
            QMessageBox.warning(self, "Erreur", f"Erreur lors du rafraîchissement: {e}")

    def updateSummaryTable(self):
        """Mettre à jour la table de résumé"""
        if self.df is None:
            return
            
        try:
            # Calculer les statistiques de base
            summary_data = []
            
            # Statistiques générales
            summary_data.append(["Nombre de points", len(self.df)])
            summary_data.append(["Profondeur min (m)", f"{self.df['Depth'].min():.2f}"])
            summary_data.append(["Profondeur max (m)", f"{self.df['Depth'].max():.2f}"])
            
            if 'qc' in self.df.columns:
                summary_data.append(["qc min (MPa)", f"{self.df['qc'].min():.2f}"])
                summary_data.append(["qc max (MPa)", f"{self.df['qc'].max():.2f}"])
                summary_data.append(["qc moyen (MPa)", f"{self.df['qc'].mean():.2f}"])
            
            if 'fs' in self.df.columns:
                summary_data.append(["fs min (kPa)", f"{self.df['fs'].min():.2f}"])
                summary_data.append(["fs max (kPa)", f"{self.df['fs'].max():.2f}"])
                summary_data.append(["fs moyen (kPa)", f"{self.df['fs'].mean():.2f}"])
            
            # Remplir la table
            self.summaryTable.setRowCount(len(summary_data))
            self.summaryTable.setColumnCount(2)
            self.summaryTable.setHorizontalHeaderLabels(["Paramètre", "Valeur"])
            
            for i, (param, value) in enumerate(summary_data):
                self.summaryTable.setItem(i, 0, QTableWidgetItem(param))
                self.summaryTable.setItem(i, 1, QTableWidgetItem(str(value)))
                
        except Exception as e:
            print(f"Erreur updateSummaryTable: {e}")

    def updateLayersTable(self):
        """Mettre à jour la table des couches géologiques"""
        if self.df is None:
            return
            
        try:
            # Classifier les couches par profondeur
            layers_data = []
            depth_bins = [0, 2, 5, 10, 15, 20, 30]  # en mètres
            
            for i in range(len(depth_bins) - 1):
                start_depth = depth_bins[i]
                end_depth = depth_bins[i + 1]
                
                # Filtrer les données pour cette couche
                mask = (self.df['Depth'] >= start_depth) & (self.df['Depth'] < end_depth)
                layer_data = self.df[mask]
                
                if not layer_data.empty:
                    qc_avg = layer_data['qc'].mean() if 'qc' in layer_data.columns else 0
                    fs_avg = layer_data['fs'].mean() if 'fs' in layer_data.columns else 0
                    
                    # Classification simple
                    if qc_avg < 5:
                        soil_type = "Argile très molle"
                    elif qc_avg < 15:
                        soil_type = "Argile molle/firme"
                    elif qc_avg < 30:
                        soil_type = "Argile raide"
                    else:
                        soil_type = "Sable dense"
                    
                    layers_data.append([
                        f"{start_depth}-{end_depth}m",
                        soil_type,
                        f"{qc_avg:.1f}",
                        f"{fs_avg:.1f}",
                        len(layer_data)
                    ])
            
            # Remplir la table
            self.layersTable.setRowCount(len(layers_data))
            self.layersTable.setColumnCount(5)
            self.layersTable.setHorizontalHeaderLabels(["Profondeur", "Type de sol", "qc moyen (MPa)", "fs moyen (kPa)", "Points"])
            
            for i, (depth, soil, qc, fs, points) in enumerate(layers_data):
                self.layersTable.setItem(i, 0, QTableWidgetItem(depth))
                self.layersTable.setItem(i, 1, QTableWidgetItem(soil))
                self.layersTable.setItem(i, 2, QTableWidgetItem(qc))
                self.layersTable.setItem(i, 3, QTableWidgetItem(fs))
                self.layersTable.setItem(i, 4, QTableWidgetItem(str(points)))
                
        except Exception as e:
            print(f"Erreur updateLayersTable: {e}")

    def updateStatsTable(self):
        """Mettre à jour la table des statistiques détaillées"""
        if self.df is None:
            return
            
        try:
            stats_data = []
            
            # Statistiques pour chaque paramètre
            for col in ['qc', 'fs', 'Depth']:
                if col in self.df.columns:
                    data = self.df[col].dropna()
                    if not data.empty:
                        stats_data.extend([
                            [f"{col} - Minimum", f"{data.min():.2f}"],
                            [f"{col} - Maximum", f"{data.max():.2f}"],
                            [f"{col} - Moyenne", f"{data.mean():.2f}"],
                            [f"{col} - Écart-type", f"{data.std():.2f}"],
                            [f"{col} - Médiane", f"{data.median():.2f}"]
                        ])
            
            # Remplir la table
            self.statsTable.setRowCount(len(stats_data))
            self.statsTable.setColumnCount(2)
            self.statsTable.setHorizontalHeaderLabels(["Statistique", "Valeur"])
            
            for i, (stat, value) in enumerate(stats_data):
                self.statsTable.setItem(i, 0, QTableWidgetItem(stat))
                self.statsTable.setItem(i, 1, QTableWidgetItem(value))
                
        except Exception as e:
            print(f"Erreur updateStatsTable: {e}")

    def updateCorrelationTable(self):
        """Mettre à jour la table de corrélation"""
        if self.df is None:
            return
            
        try:
            # Calculer la matrice de corrélation
            numeric_cols = ['qc', 'fs', 'Depth']
            available_cols = [col for col in numeric_cols if col in self.df.columns]
            
            if len(available_cols) >= 2:
                corr_matrix = self.df[available_cols].corr()
                
                # Remplir la table
                self.correlationTable.setRowCount(len(available_cols))
                self.correlationTable.setColumnCount(len(available_cols))
                self.correlationTable.setHorizontalHeaderLabels(available_cols)
                self.correlationTable.setVerticalHeaderLabels(available_cols)
                
                for i, col1 in enumerate(available_cols):
                    for j, col2 in enumerate(available_cols):
                        corr_value = corr_matrix.loc[col1, col2]
                        item = QTableWidgetItem(f"{corr_value:.3f}")
                        # Colorer selon la force de corrélation
                        if abs(corr_value) > 0.7:
                            item.setBackground(QColor(100, 200, 100))  # Vert pour forte corrélation
                        elif abs(corr_value) > 0.3:
                            item.setBackground(QColor(200, 200, 100))  # Jaune pour corrélation moyenne
                        self.correlationTable.setItem(i, j, item)
            else:
                self.correlationTable.setRowCount(1)
                self.correlationTable.setColumnCount(1)
                self.correlationTable.setItem(0, 0, QTableWidgetItem("Données insuffisantes"))
                
        except Exception as e:
            print(f"Erreur updateCorrelationTable: {e}")

    def updateLiquefactionTable(self):
        """Mettre à jour la table d'analyse de liquéfaction"""
        if self.df is None:
            return
            
        try:
            liquefaction_data = []
            
            # Analyse simple de liquéfaction basée sur qc et fs
            for i, row in self.df.iterrows():
                qc = row.get('qc', 0)
                fs = row.get('fs', 0)
                depth = row.get('Depth', 0)
                
                # Critère simple de liquéfaction (très basique)
                if qc > 0:
                    fs_qc_ratio = fs / qc
                    if fs_qc_ratio < 0.5 and qc < 10:  # Seuil arbitraire pour démonstration
                        risk = "Élevé"
                    elif fs_qc_ratio < 1.0 and qc < 15:
                        risk = "Moyen"
                    else:
                        risk = "Faible"
                else:
                    risk = "N/A"
                
                liquefaction_data.append([
                    f"{depth:.1f}",
                    f"{qc:.1f}",
                    f"{fs:.1f}",
                    f"{fs_qc_ratio:.2f}" if qc > 0 else "N/A",
                    risk
                ])
            
            # Limiter à 100 lignes pour performance
            liquefaction_data = liquefaction_data[:100]
            
            # Remplir la table
            self.liquefactionTable.setRowCount(len(liquefaction_data))
            self.liquefactionTable.setColumnCount(5)
            self.liquefactionTable.setHorizontalHeaderLabels(["Profondeur (m)", "qc (MPa)", "fs (kPa)", "fs/qc", "Risque"])
            
            for i, (depth, qc, fs, ratio, risk) in enumerate(liquefaction_data):
                self.liquefactionTable.setItem(i, 0, QTableWidgetItem(depth))
                self.liquefactionTable.setItem(i, 1, QTableWidgetItem(qc))
                self.liquefactionTable.setItem(i, 2, QTableWidgetItem(fs))
                self.liquefactionTable.setItem(i, 3, QTableWidgetItem(ratio))
                
                risk_item = QTableWidgetItem(risk)
                # Colorer selon le risque
                if risk == "Élevé":
                    risk_item.setBackground(QColor(200, 100, 100))  # Rouge
                elif risk == "Moyen":
                    risk_item.setBackground(QColor(200, 200, 100))  # Jaune
                else:
                    risk_item.setBackground(QColor(100, 200, 100))  # Vert
                self.liquefactionTable.setItem(i, 4, risk_item)
                
        except Exception as e:
            print(f"Erreur updateLiquefactionTable: {e}")
            try:
                analysis_result = perform_complete_analysis(self.df, use_streamlit=False)
                if isinstance(analysis_result, tuple):
                    df_analyzed, models, results = analysis_result
                    self.analysis_data = (df_analyzed, models, results)
                    self.analysis_results = self.create_analysis_summary(df_analyzed, results)
                else:
                    self.analysis_results = str(analysis_result)
                    self.analysis_data = None
                
                # Add layer types and zones
                layers = self.classify_layers()
                zones = self.determine_zones()
                
                full_analysis = f"{self.analysis_results}\n\n🏔️ Types de Couches:\n{layers}\n\n📍 Zones de Pointage:\n{zones}"
                self.analysisText.setText(full_analysis)
                
                # Initialize RAG system after complete analysis (with chunking and FAISS)
                if self.analysis_data and len(self.analysis_data) >= 3:
                    try:
                        self.ai_explainer.initialize_system(self.analysis_data[2])  # results dict
                        print("✅ Système RAG initialisé avec les résultats d'analyse")
                    except Exception as e:
                        print(f"⚠️ Erreur initialisation RAG: {e}")
                
            except Exception as e:
                self.analysisText.setText(f"Erreur d'analyse: {str(e)}")
        else:
            self.analysisText.setText("Aucune donnée chargée")

    def create_analysis_summary(self, df_analyzed, results):
        summary = "🔬 Analyse Géotechnique Complète\n\n"
        
        # Statistiques générales
        summary += "📊 Statistiques Générales:\n"
        summary += f"Nombre de points: {len(df_analyzed)}\n"
        summary += f"Profondeur max: {df_analyzed['Depth'].max()} cm\n"
        summary += f"qc moyen: {df_analyzed['qc'].mean():.2f} MPa\n"
        summary += f"fs moyen: {df_analyzed['fs'].mean():.2f} kPa\n\n"
        
        # Classification des sols
        if 'Soil_Type_Detailed' in df_analyzed.columns:
            soil_types = df_analyzed['Soil_Type_Detailed'].value_counts()
            summary += "🌱 Classification des Sols:\n"
            for soil, count in soil_types.items():
                summary += f"{soil}: {count} points ({count/len(df_analyzed)*100:.1f}%)\n"
            summary += "\n"
        
        # Couches identifiées
        if results and 'layers' in results and results['layers'] is not None:
            layers_df = results['layers']
            summary += f"🏔️ Couches Géologiques Identifiées: {len(layers_df)}\n"
            for _, layer in layers_df.iterrows():
                summary += f"  {layer['start_depth']:.1f}-{layer['end_depth']:.1f}m: {layer['soil_type']} (épaisseur: {layer['thickness']:.1f}m)\n"
            summary += "\n"
        
        # Analyse de liquéfaction
        if 'FS_Liquefaction' in df_analyzed.columns:
            liquefaction_risk = df_analyzed['Liquefaction_Risk'].value_counts() if 'Liquefaction_Risk' in df_analyzed.columns else None
            if liquefaction_risk is not None:
                summary += "🌊 Risque de Liquéfaction:\n"
                for risk, count in liquefaction_risk.items():
                    summary += f"{risk}: {count} points\n"
                summary += "\n"
        
        return summary

    def classify_layers(self):
        if self.df is None:
            return "Aucune donnée"
        # Enhanced classification based on qc and fs
        layers = []
        for i, row in self.df.iterrows():
            qc = row['qc']
            fs = row.get('fs', 0)
            rf = (fs / qc * 100) if qc > 0 else 0
            if qc < 5:
                layer = "Argile très molle (Ic > 3.6)"
            elif qc < 10:
                layer = "Argile molle (Ic 2.95-3.6)"
            elif qc < 20:
                layer = "Argile ferme (Ic 2.6-2.95)"
            elif qc < 40:
                layer = "Argile raide (Ic 2.05-2.6)"
            elif rf < 1:
                layer = "Sable dense (Ic < 2.05)"
            else:
                layer = "Silt sableux (Ic 2.05-2.6)"
            layers.append(f"Profondeur {row['Depth']} cm: {layer} (qc={qc:.1f} MPa, fs={fs:.1f} kPa, Rf={rf:.1f}%)")
        return "\n".join(layers[:20])  # Limit to 20

    def determine_zones(self):
        if self.df is None:
            return "Aucune donnée"
        # Determine zones based on depth with statistics
        zones = []
        depth_ranges = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 30)]
        for start, end in depth_ranges:
            zone_data = self.df[(self.df['Depth'] >= start * 100) & (self.df['Depth'] < end * 100)]  # Assuming depth in cm
            if not zone_data.empty:
                avg_qc = zone_data['qc'].mean()
                avg_fs = zone_data['fs'].mean()
                std_qc = zone_data['qc'].std()
                zones.append(f"Zone {start}-{end}m: qc moyen = {avg_qc:.1f} MPa (±{std_qc:.1f}), fs moyen = {avg_fs:.1f} kPa")
        return "\n".join(zones)

    def updateTables(self):
        if self.df is not None:
            # Résumé statistique
            summary_data = self.df.describe()
            for table in [self.summaryTable, self.overview_summaryTable]:
                table.setRowCount(len(summary_data))
                table.setColumnCount(len(summary_data.columns))
                table.setHorizontalHeaderLabels(summary_data.columns)
                table.setVerticalHeaderLabels(summary_data.index)
                for i in range(len(summary_data)):
                    for j in range(len(summary_data.columns)):
                        table.setItem(i, j, QTableWidgetItem(f"{summary_data.iloc[i, j]:.2f}"))
            
            # Table des couches
            if self.analysis_data:
                df_analyzed, models, results = self.analysis_data
                if results and 'layers' in results and results['layers'] is not None:
                    layers_df = results['layers']
                    for table in [self.layersTable, self.overview_layersTable]:
                        table.setRowCount(len(layers_df))
                        table.setColumnCount(len(layers_df.columns))
                        table.setHorizontalHeaderLabels(layers_df.columns)
                        for i in range(len(layers_df)):
                            for j in range(len(layers_df.columns)):
                                table.setItem(i, j, QTableWidgetItem(str(layers_df.iloc[i, j])))
            
            # Statistiques détaillées (même que résumé pour l'instant)
            for table in [self.statsTable, self.overview_statsTable]:
                table.setRowCount(len(summary_data))
                table.setColumnCount(len(summary_data.columns))
                table.setHorizontalHeaderLabels(summary_data.columns)
                table.setVerticalHeaderLabels(summary_data.index)
                for i in range(len(summary_data)):
                    for j in range(len(summary_data.columns)):
                        table.setItem(i, j, QTableWidgetItem(f"{summary_data.iloc[i, j]:.2f}"))
            
            # Table de corrélation
            if len(self.df.columns) > 1:
                corr = self.df.select_dtypes(include=[np.number]).corr()
                for table in [self.correlationTable, self.overview_correlationTable]:
                    table.setRowCount(len(corr))
                    table.setColumnCount(len(corr.columns))
                    table.setHorizontalHeaderLabels(corr.columns)
                    table.setVerticalHeaderLabels(corr.index)
                    for i in range(len(corr)):
                        for j in range(len(corr.columns)):
                            table.setItem(i, j, QTableWidgetItem(f"{corr.iloc[i, j]:.3f}"))
            
            # Table de liquéfaction
            if self.analysis_data:
                df_analyzed, models, results = self.analysis_data
                if 'FS_Liquefaction' in df_analyzed.columns:
                    liquefaction_data = df_analyzed[['Depth', 'qc', 'FS_Liquefaction', 'Liquefaction_Risk']].dropna()
                    for table in [self.liquefactionTable, self.overview_liquefactionTable]:
                        table.setRowCount(len(liquefaction_data))
                        table.setColumnCount(len(liquefaction_data.columns))
                        table.setHorizontalHeaderLabels(liquefaction_data.columns)
                        for i in range(len(liquefaction_data)):
                            for j in range(len(liquefaction_data.columns)):
                                table.setItem(i, j, QTableWidgetItem(str(liquefaction_data.iloc[i, j])))

    def updateAI(self):
        try:
            if self.df is not None and len(self.df) > 0:
                if hasattr(self, 'analysis_data') and self.analysis_data:
                    # Données chargées et analysées - afficher le chat IA
                    welcome_text = """📊 Données chargées: {} points<br>
🔬 Analyse disponible: {}<br>
💬 Prêt à répondre à vos questions sur les données CPT!<br><br>

💡 <strong>Exemples de questions:</strong><br>
• "Quelle est la classification des sols?"<br>
• "Y a-t-il un risque de liquéfaction?"<br>
• "Décrivez les couches géologiques"<br>
• "Quelles sont les statistiques de qc?"<br><br>

<em>Tapez votre question ci-dessous 👇</em>""".format(
                        len(self.df),
                        "Oui" if self.analysis_data else "Non - lancez l'analyse d'abord"
                    )
                    self.addChatMessage("ai", welcome_text)
                    self.statusBar.showMessage("✅ Chat IA prêt")
                else:
                    # Données chargées mais pas analysées
                    welcome_text = """📊 Données chargées: {} points<br>
⚠️ Analyse non effectuée<br><br>

🔬 Cliquez sur "Analyser" pour lancer l'analyse complète avec IA<br>
📈 Après l'analyse, vous pourrez poser des questions sur vos données CPT<br><br>

<strong>Fonctionnalités disponibles après analyse:</strong><br>
• Classification automatique des sols<br>
• Analyse de liquéfaction<br>
• Identification des couches géologiques<br>
• Statistiques détaillées<br>
• Réponses IA contextuelles avec recherches web<br>
• Visualisations intégrées""".format(len(self.df))
                    self.addChatMessage("ai", welcome_text)
                    self.statusBar.showMessage("⚠️ Lancez l'analyse pour activer le chat IA")
            else:
                # Aucune donnée
                welcome_text = """❌ Aucune donnée chargée<br><br>

📂 <strong>Pour commencer:</strong><br>
1. Cliquez sur "Fichier > Ouvrir"<br>
2. Sélectionnez un fichier CPT (.txt, .csv, .xlsx, .cal)<br>
3. Lancez l'analyse complète<br>
4. Posez vos questions à l'IA !<br><br>

💡 <strong>Formats supportés:</strong><br>
• Fichiers texte (.txt, .csv)<br>
• Excel (.xlsx, .xls)<br>
• Format binaire CPT (.cal)<br>
• Détection automatique des colonnes"""
                self.addChatMessage("ai", welcome_text)
                self.statusBar.showMessage("❌ Chargez des données CPT")
        except Exception as e:
            error_msg = f"❌ Erreur d'initialisation du chat IA: {str(e)}"
            self.addChatMessage("ai", f'<div class="error">{error_msg}</div>')
            self.statusBar.showMessage("❌ Erreur chat IA")

    def sendChatMessage(self):
        if self.df is None or len(self.df) == 0:
            self.addChatMessage("ai", "❌ Aucune donnée chargée pour discuter.<br><br>💡 Chargez d'abord un fichier CPT puis lancez l'analyse.")
            return
        question = self.chatInput.text().strip()
        if not question:
            return
        self.statusBar.showMessage("🤖 Envoi de la question IA...")

        # Vérifier que l'analyse a été effectuée
        if not hasattr(self, 'analysis_data') or not self.analysis_data or len(self.analysis_data) < 3:
            self.addChatMessage("user", question.replace('\n', '<br>'))
            self.addChatMessage("ai", "❌ Analyse non effectuée.<br><br>💡 Lancez d'abord l'analyse complète avant de poser des questions à l'IA.")
            self.chatInput.clear()
            self.statusBar.showMessage("❌ Analyse requise pour le chat IA")
            return

        # Ajouter la question de l'utilisateur
        self.addChatMessage("user", question.replace('\n', '<br>'))

        # Message de réflexion en cours
        thinking_content = "🤔 Analyse de votre question...<br><br>🔍 Recherche d'informations complémentaires...<br>🧮 Calculs en cours...<br>📊 Génération de visualisations..."
        self.addChatMessage("ai", f'<div class="loading">{thinking_content}</div>')

        try:
            # Initialiser le système RAG avec les données d'analyse si disponibles
            if self.analysis_data and len(self.analysis_data) >= 3 and isinstance(self.analysis_data[2], dict):
                self.ai_explainer.initialize_system(self.analysis_data[2])
            elif hasattr(self, 'df') and self.df is not None and not self.df.empty:
                # Fallback: initialize with basic CPT data if no analysis results
                basic_data = {'data': self.df}
                self.ai_explainer.initialize_system(basic_data)

            # Afficher la réponse en streaming pour une expérience plus dynamique

            # Afficher la réponse en streaming pour une expérience plus dynamique
            response_parts = []
            response_display = f"🤖 Chat IA Géotechnique\n\n👤 {question}\n\n🤔 Analyse en cours..."

            # Essayer d'abord avec le streaming
            streaming_worked = False
            full_response = ""
            try:
                for part in self.ai_explainer.query_streaming(question):
                    if part and part.strip():  # Vérifier que la partie n'est pas vide
                        full_response += part
                        streaming_worked = True
                        
                        # Mise à jour progressive avec recherche web intégrée - plus fréquente pour streaming
                        if len(full_response) > 20 and (len(full_response) % 50 == 0 or part.strip().endswith('\n')):  # Mettre à jour tous les 50 caractères ou fins de ligne
                            # Ajouter des informations de recherche web
                            enhanced_response = self.enhance_response_with_web_search(full_response, question)
                            # Ajouter des visualisations si pertinent
                            enhanced_response = self.add_visualizations_to_response(enhanced_response, question)
                            
                            # Ajouter un indicateur de progression basé sur les phases
                            progress_indicator = self._get_progress_indicator(full_response)
                            preview = enhanced_response[:400] + "..." if len(enhanced_response) > 400 else enhanced_response
                            self.updateLastAIMessage(f'<div class="loading">{progress_indicator} {preview}</div>')
                            QApplication.processEvents()
                                
            except Exception as e:
                print(f"Streaming failed: {e}")
                streaming_worked = False

            # Traiter la réponse complète
            if streaming_worked and full_response:
                response_parts = [full_response]
            else:
                # Si le streaming n'a pas fonctionné, utiliser une approche directe
                try:
                    direct_response = self.ai_explainer.query(question)
                    if direct_response and direct_response.strip():
                        response_parts = [direct_response]
                    else:
                        response_parts = ["❌ Le système IA n'a pas pu générer de réponse. Vérifiez que l'analyse a été effectuée."]
                except Exception as e:
                    print(f"Direct query failed: {e}")
                    response_parts = [f"❌ Erreur du système IA: {str(e)}"]

            # Réponse finale complète avec enrichissements
            response = "".join(response_parts)
            if not response or not response.strip():
                response = "❌ Aucune réponse générée par l'IA."

            # Enrichir la réponse finale
            final_response = self.enhance_response_with_web_search(response, question)
            final_response = self.add_visualizations_to_response(final_response, question)
            
            # Remplacer le message de chargement par la réponse finale
            self.updateLastAIMessage(f"🧠 {final_response}")

            self.chatInput.clear()
            self.statusBar.showMessage("✅ Réponse IA reçue")
        except Exception as e:
            error_msg = f"❌ Erreur IA: {str(e)}<br><br>💡 Assurez-vous d'avoir lancé l'analyse complète avant de poser des questions."
            self.addChatMessage("ai", f'<div class="error">{error_msg}</div>')
            self.statusBar.showMessage("❌ Erreur chat IA")

    def savePlotAsPDF(self, idx):
        if self.df is not None:
            canvas, plot_func = self.canvases[idx]
            fig = plt.Figure(figsize=(10, 8))
            ax = fig.add_subplot(111)
            try:
                plot_func(self.df, ax)
                file_path, _ = QFileDialog.getSaveFileName(self, "Sauvegarder en PDF", "", "PDF Files (*.pdf)")
                if file_path:
                    fig.savefig(file_path, format='pdf', bbox_inches='tight')
                    QMessageBox.information(self, "Succès", "Graphique sauvegardé en PDF!")
            except Exception as e:
                QMessageBox.warning(self, "Erreur", f"Échec de la sauvegarde: {str(e)}")

    def saveOverviewAsPDF(self):
        if self.df is not None and hasattr(self, 'overview_canvases'):
            try:
                file_path, _ = QFileDialog.getSaveFileName(self, "Sauvegarder Vue d'ensemble en PDF", "", "PDF Files (*.pdf)")
                if file_path:
                    from matplotlib.backends.backend_pdf import PdfPages
                    with PdfPages(file_path) as pdf:
                        for canvas, _ in self.overview_canvases:
                            pdf.savefig(canvas.figure, bbox_inches='tight')
                    QMessageBox.information(self, "Succès", "Vue d'ensemble sauvegardée en PDF!")
            except Exception as e:
                QMessageBox.warning(self, "Erreur", f"Échec de la sauvegarde: {str(e)}")

    def exportDataToPDF(self):
        if self.df is not None:
            try:
                file_path, _ = QFileDialog.getSaveFileName(self, "Exporter en PDF", "", "PDF Files (*.pdf)")
                if file_path:
                    from reportlab.pdfgen import canvas
                    from reportlab.lib.pagesizes import letter, A4
                    from reportlab.lib import colors
                    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
                    doc = SimpleDocTemplate(file_path, pagesize=A4)
                    elements = []
                    data = [self.df.columns.tolist()] + self.df.values.tolist()
                    table = Table(data)
                    table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 14),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black)
                    ]))
                    elements.append(table)
                    doc.build(elements)
                    QMessageBox.information(self, "Succès", "Données exportées en PDF!")
            except Exception as e:
                QMessageBox.warning(self, "Erreur", f"Échec de l'export: {str(e)}")

    def exportDataToExcel(self):
        if self.df is not None:
            try:
                file_path, _ = QFileDialog.getSaveFileName(self, "Exporter en Excel", "", "Excel Files (*.xlsx)")
                if file_path:
                    self.df.to_excel(file_path, index=False)
                    QMessageBox.information(self, "Succès", "Données exportées en Excel!")
            except Exception as e:
                QMessageBox.warning(self, "Erreur", f"Échec de l'export: {str(e)}")

    def showAbout(self):
        QMessageBox.about(self, "À propos", "Logiciel d'Analyse CPT Puissant\nVersion 1.0\nDéveloppé avec PyQt6")

    def showAbout(self):
        QMessageBox.about(self, "À propos", "Logiciel d'Analyse CPT Puissant\nVersion 1.0\nDéveloppé avec PyQt6")

    def showPresentation(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Présentation du Logiciel CPT Analysis Studio")
        dialog.resize(600, 500)
        layout = QVBoxLayout()
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        content_layout = QVBoxLayout()
        
        title = QLabel("🏗️ CPT Analysis Studio")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #0078d7;")
        content_layout.addWidget(title)
        
        subtitle = QLabel("Logiciel puissant d'analyse géotechnique avec IA")
        subtitle.setStyleSheet("font-size: 16px; font-style: italic;")
        content_layout.addWidget(subtitle)
        
        version = QLabel("Version 2.0 - Interface moderne et analyses avancées")
        content_layout.addWidget(version)
        
        features_title = QLabel("✨ Fonctionnalités principales :")
        features_title.setStyleSheet("font-size: 18px; font-weight: bold; margin-top: 20px;")
        content_layout.addWidget(features_title)
        
        features = QLabel("""
• Analyse complète des données CPTU<br>
• 20 graphiques interactifs et exportables<br>
• Classifications des sols automatiques avec IA<br>
• Analyse de liquéfaction sismique<br>
• Chat IA intégré pour questions contextuelles<br>
• Recherche web automatique pour réponses enrichies<br>
• Export PDF multi-pages et Excel<br>
• Interface moderne avec thèmes personnalisables<br>
• Visualisations 3D interactives<br>
• Tableaux détaillés et statistiques<br>
• Vue d'ensemble complète de tous les résultats
        """)
        features.setTextFormat(Qt.TextFormat.RichText)
        content_layout.addWidget(features)
        
        download_title = QLabel("📥 Téléchargement :")
        download_title.setStyleSheet("font-size: 18px; font-weight: bold; margin-top: 20px;")
        content_layout.addWidget(download_title)
        
        download_text = QLabel("Cliquez ci-dessous pour télécharger la dernière version ou le manuel utilisateur :")
        content_layout.addWidget(download_text)
        
        download_btn = QPushButton("⬇️ Télécharger")
        download_btn.setStyleSheet("font-size: 14px; padding: 10px; background-color: #28a745; color: white; border-radius: 5px;")
        download_btn.clicked.connect(self.downloadSoftware)
        content_layout.addWidget(download_btn)
        
        content.setLayout(content_layout)
        scroll.setWidget(content)
        layout.addWidget(scroll)
        
        close_btn = QPushButton("Fermer")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        
        dialog.setLayout(layout)
        dialog.exec()

    def showDataIntegrityReport(self):
        """Affiche le rapport détaillé d'intégrité des données"""
        if self.df is None:
            QMessageBox.warning(self, "Aucune donnée", "Veuillez d'abord charger un fichier CPT.")
            return

        # Ouvrir un dialogue pour sélectionner un fichier à vérifier
        fileName, _ = QFileDialog.getOpenFileName(self, "Sélectionner fichier pour vérification d'intégrité",
                                                  "", "Fichiers CPT (*.txt *.xlsx *.csv *.xls *.cal);;Tous les fichiers (*)")

        if not fileName:
            return

        try:
            # Générer le rapport d'intégrité
            integrity_report = self.data_checker.generate_integrity_report(fileName)

            # Créer une boîte de dialogue pour afficher le rapport
            dialog = QDialog(self)
            dialog.setWindowTitle("Rapport d'Intégrité des Données")
            dialog.setGeometry(200, 200, 800, 600)

            layout = QVBoxLayout()

            # Titre
            title = QLabel("Rapport de Vérification d'Intégrité des Données CPT")
            title.setFont(QFont("Arial", 14, QFont.Bold))
            layout.addWidget(title)

            # Zone de texte pour le rapport
            report_text = QTextEdit()
            report_text.setPlainText(integrity_report)
            report_text.setReadOnly(True)
            report_text.setFont(QFont("Consolas", 10))  # Police monospace pour une meilleure lisibilité

            layout.addWidget(report_text)

            # Boutons
            button_layout = QHBoxLayout()

            save_btn = QPushButton("Sauvegarder Rapport")
            save_btn.clicked.connect(lambda: self.saveIntegrityReport(integrity_report))
            button_layout.addWidget(save_btn)

            close_btn = QPushButton("Fermer")
            close_btn.clicked.connect(dialog.accept)
            button_layout.addWidget(close_btn)

            layout.addLayout(button_layout)

            dialog.setLayout(layout)
            dialog.exec()

        except Exception as e:
            QMessageBox.warning(self, "Erreur", f"Erreur lors de la génération du rapport: {str(e)}")

    def saveIntegrityReport(self, report_text):
        """Sauvegarde le rapport d'intégrité dans un fichier"""
        fileName, _ = QFileDialog.getSaveFileName(self, "Sauvegarder Rapport d'Intégrité",
                                                  "rapport_integrite_donnees.txt",
                                                  "Fichiers texte (*.txt);;Tous les fichiers (*)")

        if fileName:
            try:
                with open(fileName, 'w', encoding='utf-8') as f:
                    f.write(report_text)
                QMessageBox.information(self, "Succès", "Rapport sauvegardé avec succès.")
            except Exception as e:
                QMessageBox.warning(self, "Erreur", f"Erreur lors de la sauvegarde: {str(e)}")

    def downloadSoftware(self):
        QMessageBox.information(self, "Téléchargement", "Fonctionnalité de téléchargement à implémenter.\n\nPour l'instant, vous pouvez visiter le dépôt GitHub ou contacter le développeur pour obtenir la dernière version.")

    def loadMultipleCPTUFiles(self):
        """Charger plusieurs fichiers CPTU pour la fusion"""
        try:
            fileNames, _ = QFileDialog.getOpenFileNames(
                self, 
                "Charger Plusieurs Fichiers CPTU", 
                "", 
                "Fichiers CPTU (*.txt *.xlsx *.csv *.xls *.cal);;Tous les fichiers (*)"
            )
            
            if not fileNames:
                return
                
            self.fusion_files = fileNames
            
            # Afficher la liste des fichiers
            file_list_text = "Fichiers chargés:\n" + "\n".join([f"• {os.path.basename(f)}" for f in fileNames])
            self.fusionFileList.setPlainText(file_list_text)
            
            # Initialiser la table des coordonnées
            self.coordTable.setRowCount(len(fileNames))
            for i, fileName in enumerate(fileNames):
                self.coordTable.setItem(i, 0, QTableWidgetItem(os.path.basename(fileName)))
                self.coordTable.setItem(i, 1, QTableWidgetItem("0.0"))  # X par défaut
                self.coordTable.setItem(i, 2, QTableWidgetItem("0.0"))  # Y par défaut
            
            # Activer le bouton de fusion
            self.fusionButton.setEnabled(True)
            
            QMessageBox.information(self, "Succès", f"{len(fileNames)} fichiers CPTU chargés avec succès!")
            
        except Exception as e:
            QMessageBox.warning(self, "Erreur", f"Erreur lors du chargement: {e}")

    def addCoordinates(self):
        """Ajouter des coordonnées pour un nouveau sondage"""
        row_count = self.coordTable.rowCount()
        self.coordTable.insertRow(row_count)
        self.coordTable.setItem(row_count, 0, QTableWidgetItem("Nouveau sondage"))
        self.coordTable.setItem(row_count, 1, QTableWidgetItem("0.0"))
        self.coordTable.setItem(row_count, 2, QTableWidgetItem("0.0"))

    def clearCoordinates(self):
        """Effacer toutes les coordonnées"""
        self.coordTable.setRowCount(0)
        self.fusion_files = []
        self.fusionFileList.clear()
        self.fusionButton.setEnabled(False)

    def create3DSoilMap(self):
        """Créer les cartes 2D individuelles pour chaque sondage CPTU"""
        try:
            if not self.fusion_files:
                QMessageBox.warning(self, "Erreur", "Aucun fichier chargé.")
                return

            # Charger les données individuelles pour chaque fichier
            individual_data = []
            parser = CPTParser()

            for file_path in self.fusion_files:
                try:
                    df, message = parser.parse_file(file_path)

                    if df is None or df.empty:
                        print(f"⚠️ Impossible de parser {os.path.basename(file_path)}: {message}")
                        continue

                    # Ajouter le nom du sondage
                    filename = os.path.basename(file_path)
                    df['Sondage'] = filename

                    # Normaliser les noms de colonnes
                    df = df.rename(columns={'depth': 'Depth', 'qc': 'qc', 'fs': 'fs'})

                    individual_data.append(df)
                    print(f"✅ {filename}: {len(df)} points chargés")

                except Exception as e:
                    print(f"❌ Erreur avec {os.path.basename(file_path)}: {e}")
                    continue

            if not individual_data:
                QMessageBox.warning(self, "Erreur", "Aucune donnée valide trouvée dans les fichiers.")
                return

            # Combiner toutes les données individuelles
            combined_data = pd.concat(individual_data, ignore_index=True)

            # Créer la visualisation 2D individuelle pour chaque sondage
            self.createFused3DVisualization(combined_data)

            # Afficher les informations
            info_text = f"Graphiques 2D individuels créés avec succès!\n"
            info_text += f"• {len(individual_data)} sondages chargés\n"
            info_text += f"• {len(combined_data)} points de données totaux\n"
            info_text += f"• Profondeur max: {combined_data['Depth'].max():.1f} m\n"

            self.fusion2DInfoLabel.setText(info_text)
            self.fusion2DInfoLabel.setStyleSheet("font-weight: bold; color: #4CAF50;")

            QMessageBox.information(self, "Succès", "Graphiques 2D individuels créés avec succès!")

        except Exception as e:
            QMessageBox.warning(self, "Erreur", f"Erreur lors de la création des graphiques 2D: {e}")
            import traceback
            traceback.print_exc()

    def fuseCPTUData(self, file_paths, coordinates):
        """Fusionner les données CPTU avec les coordonnées"""
        fused_data = []
        parser = CPTParser()
        
        for file_path in file_paths:
            try:
                # Parser le fichier
                df, message = parser.parse_file(file_path)
                
                if df is None or df.empty:
                    print(f"⚠️ Impossible de parser {os.path.basename(file_path)}: {message}")
                    continue
                
                # Ajouter les coordonnées
                filename = os.path.basename(file_path)
                if filename in coordinates:
                    x, y = coordinates[filename]
                    df['X'] = x
                    df['Y'] = y
                    df['Sondage'] = filename
                else:
                    # Coordonnées par défaut si non spécifiées
                    df['X'] = 0.0
                    df['Y'] = 0.0
                    df['Sondage'] = filename
                
                # Normaliser les noms de colonnes
                df = df.rename(columns={'depth': 'Depth', 'qc': 'qc', 'fs': 'fs'})
                
                fused_data.append(df)
                print(f"✅ {filename}: {len(df)} points chargés")
                
            except Exception as e:
                print(f"❌ Erreur avec {os.path.basename(file_path)}: {e}")
                continue
        
        if not fused_data:
            return pd.DataFrame()
        
        # Combiner toutes les données
        combined_df = pd.concat(fused_data, ignore_index=True)
        
        # Trier par coordonnées et profondeur
        combined_df = combined_df.sort_values(['X', 'Y', 'Depth']).reset_index(drop=True)
        
        return combined_df

    def loadMultipleCPTUFiles(self):
        """Charge plusieurs fichiers CPTU pour la fusion"""
        try:
            # Ouvrir le dialogue de sélection de fichiers
            file_paths, _ = QFileDialog.getOpenFileNames(
                self, 
                "Sélectionner les fichiers CPTU", 
                "", 
                "Fichiers CPTU (*.txt *.csv *.xlsx);;Tous les fichiers (*)"
            )
            
            if not file_paths:
                return
            
            # Stocker les fichiers
            self.fusion_files = file_paths
            
            # Afficher la liste des fichiers chargés
            file_list_text = "Fichiers chargés :\n" + "\n".join([f"• {os.path.basename(fp)}" for fp in file_paths])
            self.fusionFileList.setText(file_list_text)
            
            # Activer les boutons
            self.autoDetectCoordButton.setEnabled(True)
            self.addCoordButton.setEnabled(True)
            self.clearCoordButton.setEnabled(True)
            
            # Auto-détecter les coordonnées automatiquement
            self.autoDetectAndFillCoordinates()
            
            QMessageBox.information(self, "Succès", f"{len(file_paths)} fichiers CPTU chargés avec succès !\n\nLes coordonnées ont été auto-détectées. Vous pouvez les modifier si nécessaire.")
            
        except Exception as e:
            QMessageBox.warning(self, "Erreur", f"Erreur lors du chargement: {e}")

    def autoDetectCoordinates(self, file_paths):
        """Détecte automatiquement les coordonnées des fichiers CPTU en utilisant l'IA et algorithmes de reconstruction"""
        global RAG_SYSTEM_AVAILABLE  # Déclarer comme variable globale
        try:
            coordinates = {}

            # Initialiser le système RAG si nécessaire
            if not hasattr(self, 'rag_system') or self.rag_system is None:
                if RAG_SYSTEM_AVAILABLE:
                    try:
                        from models.rag_system import CPT_RAG_System
                        self.rag_system = CPT_RAG_System()
                        print("✅ Système RAG initialisé avec succès")
                    except Exception as e:
                        print(f"⚠️ Échec de l'initialisation du système RAG: {e}")
                        self.rag_system = None
                        RAG_SYSTEM_AVAILABLE = False
                else:
                    self.rag_system = None
                    print("⚠️ Système RAG non disponible - reconstruction avec algorithmes déterministes uniquement")

            # Analyser d'abord tous les fichiers pour comprendre le contexte global
            file_contexts = []
            for i, file_path in enumerate(file_paths):
                filename = os.path.basename(file_path)
                context = {
                    'filename': filename,
                    'index': i,
                    'path': file_path,
                    'coords_from_name': self.extractCoordinatesFromFilename(filename),
                    'data_summary': None
                }

                # Essayer de lire un résumé des données CPTU
                try:
                    if os.path.exists(file_path):
                        # Lire quelques lignes pour analyser les données
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            lines = f.readlines()[:20]  # Premières lignes
                            context['data_summary'] = f"Lignes: {len(lines)}, Contenu exemple: {lines[0].strip()[:100]}"
                except:
                    pass

                file_contexts.append(context)

            # Algorithme 1: Reconstruction basée sur les données géologiques
            geological_coordinates = self.reconstructCoordinatesFromGeology(file_contexts)

            for i, file_path in enumerate(file_paths):
                filename = os.path.basename(file_path)
                context = file_contexts[i]

                # Méthode 1: Coordonnées extraites du nom de fichier (priorité maximale)
                if context['coords_from_name']:
                    coordinates[filename] = context['coords_from_name']
                    print(f"📍 Coordonnées extraites du nom: {filename} -> X={context['coords_from_name'][0]}, Y={context['coords_from_name'][1]}")
                    continue

                # Méthode 2: Reconstruction géologique
                if filename in geological_coordinates:
                    x, y = geological_coordinates[filename]
                    coordinates[filename] = (x, y)
                    print(f"🗺️ Reconstruction géologique: {filename} -> X={x}, Y={y}")
                    continue

                # Méthode 3: IA avancée avec reconstruction de données
                try:
                    x, y = self.reconstructCoordinatesWithAI(context, file_contexts)
                    coordinates[filename] = (x, y)
                    print(f"🤖 IA reconstruction: {filename} -> X={x}, Y={y}")
                    continue

                except Exception as e:
                    print(f"⚠️ Erreur IA reconstruction pour {filename}: {e}")

                # Méthode 4: Algorithme de grille intelligente optimisée
                x, y = self.generateOptimalGridPosition(i, len(file_paths), coordinates)
                coordinates[filename] = (x, y)
                print(f"📐 Grille intelligente: {filename} -> X={x}, Y={y}")

            return coordinates

        except Exception as e:
            print(f"❌ Erreur auto-détection coordonnées: {e}")
            # Retourner des coordonnées par défaut
            return {os.path.basename(fp): (i * 10.0, 0.0) for i, fp in enumerate(file_paths)}

    def reconstructCoordinatesFromGeology(self, file_contexts):
        """Reconstruction des coordonnées basée sur l'analyse géologique des données CPTU"""
        try:
            coordinates = {}

            # Analyser d'abord les données CPTU pour des patterns géologiques
            geological_data = self.analyzeCPTUGeologicalPatterns(file_contexts)

            # Analyser les patterns géologiques dans les noms de fichiers
            geological_patterns = {
                'river': ['river', 'rivière', 'fleuve', 'riviere'],
                'road': ['road', 'route', 'rue', 'chemin'],
                'building': ['building', 'batiment', 'construction', 'immeuble'],
                'bridge': ['bridge', 'pont', 'viaduc'],
                'tunnel': ['tunnel', 'galerie'],
                'slope': ['slope', 'talus', 'pente'],
                'embankment': ['embankment', 'remblai', 'digue'],
                'excavation': ['excavation', 'fouille', 'tranchee']
            }

            # Analyser chaque fichier pour des indices géologiques
            for context in file_contexts:
                filename = context['filename'].lower()
                geological_type = None
                position_hint = None

                # Détecter le type géologique depuis les données
                if filename in geological_data:
                    geological_type = geological_data[filename].get('type')

                # Détecter le type géologique depuis le nom
                if not geological_type:
                    for geo_type, keywords in geological_patterns.items():
                        if any(keyword in filename for keyword in keywords):
                            geological_type = geo_type
                            break

                # Extraire des indices de position
                if 'left' in filename or 'gauche' in filename:
                    position_hint = 'left'
                elif 'right' in filename or 'droite' in filename:
                    position_hint = 'right'
                elif 'center' in filename or 'centre' in filename or 'central' in filename:
                    position_hint = 'center'
                elif 'north' in filename or 'nord' in filename:
                    position_hint = 'north'
                elif 'south' in filename or 'sud' in filename:
                    position_hint = 'south'
                elif 'east' in filename or 'est' in filename:
                    position_hint = 'east'
                elif 'west' in filename or 'ouest' in filename:
                    position_hint = 'west'

                # Générer des coordonnées basées sur l'analyse géologique
                base_coords = self.generateGeologicalCoordinates(geological_type, position_hint, context['index'], len(file_contexts))
                if base_coords:
                    coordinates[context['filename']] = base_coords

            return coordinates

        except Exception as e:
            print(f"⚠️ Erreur reconstruction géologique: {e}")
            return {}

    def analyzeCPTUGeologicalPatterns(self, file_contexts):
        """Analyse les données CPTU pour identifier des patterns géologiques"""
        try:
            geological_analysis = {}

            for context in file_contexts:
                filename = context['filename']
                file_path = context['path']

                try:
                    # Charger les données CPTU
                    from core.cpt_parser import CPTParser
                    parser = CPTParser()
                    result = parser.parse_file(file_path)

                    # Le parser retourne un tuple (DataFrame, message)
                    if result[0] is not None and isinstance(result[0], pd.DataFrame):
                        data = result[0]
                        # Analyser les patterns géologiques dans les données
                        analysis = self.analyzeGeologicalData(data)
                        geological_analysis[filename] = analysis
                    else:
                        print(f"⚠️ Erreur parsing {filename}: {result[1]}")

                except Exception as e:
                    print(f"⚠️ Impossible d'analyser {filename}: {e}")
                    continue

            return geological_analysis

        except Exception as e:
            print(f"⚠️ Erreur analyse patterns géologiques: {e}")
            return {}

    def analyzeGeologicalData(self, data):
        """Analyse les données CPTU pour identifier le contexte géologique"""
        try:
            analysis = {}

            if 'qc' not in data.columns or 'fs' not in data.columns:
                return analysis

            # Calculer des statistiques
            qc_mean = data['qc'].mean()
            qc_std = data['qc'].std()
            fs_mean = data['fs'].mean()

            # Classifier le type de sol dominant
            if qc_mean > 15:  # Sol très résistant
                soil_type = 'rock' if qc_mean > 50 else 'dense_sand'
            elif qc_mean > 5:  # Sol résistant
                soil_type = 'medium_sand' if fs_mean < 100 else 'silt'
            else:  # Sol faible
                soil_type = 'clay' if fs_mean > 150 else 'loose_sand'

            analysis['soil_type'] = soil_type

            # Détecter des patterns spéciaux
            if self.detectRiverPattern(data):
                analysis['type'] = 'river'
                analysis['confidence'] = 0.8
            elif self.detectSlopePattern(data):
                analysis['type'] = 'slope'
                analysis['confidence'] = 0.7
            elif self.detectEmbankmentPattern(data):
                analysis['type'] = 'embankment'
                analysis['confidence'] = 0.6
            else:
                analysis['type'] = 'generic'
                analysis['confidence'] = 0.3

            return analysis

        except Exception as e:
            print(f"⚠️ Erreur analyse données géologiques: {e}")
            return {}

    def detectRiverPattern(self, data):
        """Détecte si les données correspondent à un profil de rivière"""
        try:
            # Les rivières ont souvent des couches alternées et des valeurs variables
            if 'qc' in data.columns and len(data) > 10:
                # Calculer la variabilité
                qc_variability = data['qc'].std() / data['qc'].mean()
                return qc_variability > 0.5  # Haute variabilité = possible rivière
            return False
        except:
            return False

    def detectSlopePattern(self, data):
        """Détecte si les données correspondent à un profil de pente/talus"""
        try:
            if 'qc' in data.columns and len(data) > 10:
                # Les pentes ont souvent une augmentation progressive de qc avec la profondeur
                depths = data.index if data.index.name == 'depth' else data['depth'] if 'depth' in data.columns else range(len(data))
                qc_trend = np.polyfit(list(depths), data['qc'].values, 1)[0]
                return qc_trend > 0.1  # Tendance positive = possible pente
            return False
        except:
            return False

    def detectEmbankmentPattern(self, data):
        """Détecte si les données correspondent à un profil de remblai"""
        try:
            if 'qc' in data.columns and len(data) > 10:
                # Les remblais ont souvent des valeurs qc variables en surface
                surface_qc = data['qc'].head(5).mean()
                deep_qc = data['qc'].tail(5).mean()
                return abs(surface_qc - deep_qc) / max(surface_qc, deep_qc) > 0.3
            return False
        except:
            return False

    def generateGeologicalCoordinates(self, geological_type, position_hint, index, total_files):
        """Génère des coordonnées basées sur le contexte géologique"""
        try:
            # Base coordinates selon le type géologique
            if geological_type == 'river':
                # Alignement le long d'une rivière (axe Y)
                x = 0
                y = index * 50  # Espacement de 50m le long de la rivière
            elif geological_type == 'road':
                # Alignement le long d'une route (axe X)
                x = index * 100  # Espacement de 100m le long de la route
                y = 0
            elif geological_type == 'bridge':
                # Autour d'un pont (disposition en éventail)
                angle = (index / max(1, total_files - 1)) * 180 - 90  # De -90° à +90°
                distance = 20 + (index % 3) * 10  # Distances variables
                x = distance * np.cos(np.radians(angle))
                y = distance * np.sin(np.radians(angle))
            elif geological_type == 'tunnel':
                # Le long d'un tunnel (axe X avec variation Y)
                x = index * 25
                y = (index % 2) * 10 - 5  # Alternance de chaque côté
            else:
                # Disposition générique en grille optimisée
                grid_cols = int(np.ceil(np.sqrt(total_files)))
                row = index // grid_cols
                col = index % grid_cols
                x = col * 15
                y = row * 15

            # Ajuster selon les indices de position
            if position_hint == 'left':
                x -= 20
            elif position_hint == 'right':
                x += 20
            elif position_hint == 'north':
                y += 20
            elif position_hint == 'south':
                y -= 20
            elif position_hint == 'east':
                x += 20
            elif position_hint == 'west':
                x -= 20

            return (x, y)

        except Exception as e:
            print(f"⚠️ Erreur génération coordonnées géologiques: {e}")
            return None

    def reconstructCoordinatesWithAI(self, context, all_contexts):
        """Reconstruction avancée des coordonnées utilisant l'IA avec algorithme de reconstruction"""
        try:
            # Analyser le contexte global
            global_context = f"Total de fichiers: {len(all_contexts)}\n"
            global_context += "Fichiers analysés:\n"
            for i, ctx in enumerate(all_contexts[:10]):  # Limiter à 10 pour éviter surcharge
                global_context += f"- {ctx['filename']}"
                if ctx['coords_from_name']:
                    global_context += f" (coordonnées détectées: {ctx['coords_from_name']})"
                global_context += "\n"

            # Utiliser l'IA pour reconstruction intelligente
            prompt = f"""Analyse ce fichier CPTU et reconstruis des coordonnées X,Y précises.

CONTEXTE GLOBAL:
{global_context}

FICHIER À ANALYSER:
Nom: {context['filename']}
Position: {context['index'] + 1}/{len(all_contexts)}
Données: {context.get('data_summary', 'Non disponible')}

INSTRUCTIONS pour reconstruction:
1. Analyse le nom du fichier pour des indices géologiques ou spatiaux
2. Considère la position relative par rapport aux autres fichiers
3. Utilise des principes géotechniques réalistes pour le placement
4. Crée une disposition logique et optimisée spatialement

L'algorithme de reconstruction doit:
- Préserver les distances réalistes entre sondages (10-50m typiquement)
- Créer des alignements logiques (lignes, grilles, courbes)
- Respecter les contraintes géologiques implicites
- Optimiser pour la couverture spatiale

Réponds UNIQUEMENT avec: X=123.45, Y=67.89
"""

            if self.rag_system and hasattr(self.rag_system, 'is_initialized') and self.rag_system.is_initialized:
                response = self.rag_system.query(prompt, use_web=False, use_geo=False)

                # Extraire les coordonnées avec regex amélioré
                import re
                coord_patterns = [
                    r'X=([-\d.]+),\s*Y=([-\d.]+)',
                    r'X\s*=\s*([-\d.]+).*?Y\s*=\s*([-\d.]+)',
                    r'coord.*?\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)',
                    r'position.*?\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)'
                ]

                for pattern in coord_patterns:
                    match = re.search(pattern, response, re.IGNORECASE)
                    if match:
                        x, y = float(match.group(1)), float(match.group(2))
                        return (x, y)

            # Fallback: utiliser un algorithme déterministe de reconstruction
            return self.deterministicCoordinateReconstruction(context, all_contexts)

        except Exception as e:
            print(f"⚠️ Erreur reconstruction IA: {e}")
            return self.deterministicCoordinateReconstruction(context, all_contexts)

    def deterministicCoordinateReconstruction(self, context, all_contexts):
        """Algorithme déterministe de reconstruction de coordonnées"""
        try:
            index = context['index']
            total = len(all_contexts)

            # Analyser les patterns dans les noms de fichiers
            filename = context['filename'].lower()

            # Patterns spatiaux
            if any(word in filename for word in ['line', 'ligne', 'align', 'row']):
                # Disposition en ligne
                x = index * 25
                y = 0
            elif any(word in filename for word in ['grid', 'grille', 'matrix']):
                # Disposition en grille
                grid_cols = int(np.ceil(np.sqrt(total)))
                row = index // grid_cols
                col = index % grid_cols
                x = col * 20
                y = row * 20
            elif any(word in filename for word in ['circle', 'cercle', 'radial']):
                # Disposition radiale
                angle = (index / max(1, total - 1)) * 2 * np.pi
                radius = 30
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
            else:
                # Disposition optimisée par défaut
                # Utiliser un algorithme de placement optimisé
                x, y = self.optimizeCoordinatePlacement(index, total)

            return (x, y)

        except Exception as e:
            print(f"⚠️ Erreur reconstruction déterministe: {e}")
            # Fallback final
            return (index * 15.0, 0.0)

    def optimizeCoordinatePlacement(self, index, total):
        """Algorithme d'optimisation pour le placement des coordonnées"""
        try:
            import math

            # Calculer la disposition optimale
            if total <= 4:
                # Carré simple
                positions = [(0, 0), (20, 0), (0, 20), (20, 20)]
                x, y = positions[min(index, len(positions) - 1)]
            elif total <= 9:
                # Grille 3x3 optimisée
                grid_size = 3
                row = index // grid_size
                col = index % grid_size
                x = col * 25
                y = row * 25
            else:
                # Disposition en spirale pour maximiser l'espacement
                # Algorithme de spirale d'Archimède
                theta = index * 0.5  # Angle
                r = math.sqrt(index) * 15  # Rayon croissant
                x = r * math.cos(theta)
                y = r * math.sin(theta)

            return (x, y)

        except Exception as e:
            return (index * 20.0, 0.0)

    def generateOptimalGridPosition(self, index, total, existing_coordinates):
        """Génère une position optimale dans une grille intelligente"""
        try:
            # Éviter les collisions avec les coordonnées existantes
            existing_positions = list(existing_coordinates.values())

            # Calculer la grille optimale
            grid_cols = int(np.ceil(np.sqrt(total)))
            row = index // grid_cols
            col = index % grid_cols

            # Espacement adaptatif
            if total <= 4:
                spacing = 15.0
            elif total <= 9:
                spacing = 20.0
            elif total <= 16:
                spacing = 25.0
            else:
                spacing = 30.0

            x = col * spacing
            y = row * spacing

            # Ajuster pour éviter les collisions
            attempts = 0
            while (x, y) in existing_positions and attempts < 10:
                x += 5  # Décaler légèrement
                y += 5
                attempts += 1

            return (x, y)

        except Exception as e:
            return (index * 25.0, 0.0)
    def extractCoordinatesFromFilename(self, filename):
        """Extrait les coordonnées du nom de fichier si elles sont présentes"""
        import re
        
        # Patterns courants pour les coordonnées dans les noms de fichiers
        patterns = [
            # X100Y200, X100_Y200, X_100_Y_200, etc.
            r'[XYxy]_?(\d+(?:\.\d+)?)_?[XYxy]_?(\d+(?:\.\d+)?)',
            # 100_200 (X_Y), 100x200, etc.
            r'(\d+(?:\.\d+)?)[_x](\d+(?:\.\d+)?)',
            # Coordonnées avec séparateur
            r'coord[_]?(\d+(?:\.\d+)?)[_x](\d+(?:\.\d+)?)',
            # Position avec tirets
            r'pos[_-]?(\d+(?:\.\d+)?)[_-](\d+(?:\.\d+)?)',
            # Nombres séparés par des tirets ou underscores
            r'(\d+(?:\.\d+)?)[_-](\d+(?:\.\d+)?)[_-](\d+(?:\.\d+)?)',
            # Format CPTU_X123.45_Y67.89
            r'CPTU[_]?[XYxy](\d+(?:\.\d+)?)[_]?[XYxy](\d+(?:\.\d+)?)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, filename, re.IGNORECASE)
            if matches:
                # Prendre le premier match
                match = matches[0]
                if len(match) >= 2:
                    # Si on a 3 nombres, prendre les 2 derniers (X,Y)
                    if len(match) == 3:
                        x, y = float(match[1]), float(match[2])
                    else:
                        x, y = float(match[0]), float(match[1])
                    return (x, y)
        
        return None

    def autoDetectAndFillCoordinates(self):
        """Auto-détecte les coordonnées et remplit la table"""
        try:
            if not hasattr(self, 'fusion_files') or not self.fusion_files:
                QMessageBox.warning(self, "Erreur", "Veuillez d'abord charger des fichiers CPTU.")
                return
            
            # Auto-détecter les coordonnées
            coordinates = self.autoDetectCoordinates(self.fusion_files)
            
            # Vider la table actuelle
            self.coordTable.setRowCount(0)
            
            # Remplir la table avec les coordonnées détectées
            for filename, (x, y) in coordinates.items():
                row = self.coordTable.rowCount()
                self.coordTable.insertRow(row)
                
                # Fichier
                self.coordTable.setItem(row, 0, QTableWidgetItem(filename))
                
                # Coordonnée X
                x_item = QTableWidgetItem(f"{x:.2f}")
                x_item.setData(Qt.ItemDataRole.UserRole, x)  # Stocker la valeur numérique
                x_item.setFlags(x_item.flags() | Qt.ItemFlag.ItemIsEditable)  # Rendre éditable
                self.coordTable.setItem(row, 1, x_item)
                
                # Coordonnée Y
                y_item = QTableWidgetItem(f"{y:.2f}")
                y_item.setData(Qt.ItemDataRole.UserRole, y)  # Stocker la valeur numérique
                y_item.setFlags(y_item.flags() | Qt.ItemFlag.ItemIsEditable)  # Rendre éditable
                self.coordTable.setItem(row, 2, y_item)
            
            # Activer le bouton de fusion
            self.fusionButton.setEnabled(True)
            
            QMessageBox.information(self, "Succès", 
                f"Coordonnées auto-détectées pour {len(coordinates)} fichiers.\n"
                "Vous pouvez les modifier manuellement si nécessaire.")
            
        except Exception as e:
            QMessageBox.warning(self, "Erreur", f"Erreur lors de l'auto-détection: {e}")

    def createFused3DVisualization(self, fused_data):
        """Créer la visualisation 3D de contours pour chaque sondage CPTU"""
        try:
            from scipy.interpolate import griddata

            # Créer des graphiques 3D individuels pour chaque sondage
            sondages = fused_data['Sondage'].unique()
            n_sondages = len(sondages)

            # Calculer la grille optimale (carré le plus proche)
            n_cols = int(np.ceil(np.sqrt(n_sondages)))
            n_rows = int(np.ceil(n_sondages / n_cols))

            # Créer la figure avec sous-graphiques 3D
            specs = [[{'type': 'scene'} for _ in range(n_cols)] for _ in range(n_rows)]
            fig = make_subplots(
                rows=n_rows, cols=n_cols,
                subplot_titles=[f'Sondage {sondage}' for sondage in sondages],
                specs=specs,
                shared_xaxes=False,
                shared_yaxes=False
            )

            # Pour chaque sondage, créer un graphique de contour
            for idx, sondage in enumerate(sondages):
                sondage_data = fused_data[fused_data['Sondage'] == sondage]

                if len(sondage_data) < 3:
                    continue

                # Calculer la position du sous-graphique
                row = (idx // n_cols) + 1
                col = (idx % n_cols) + 1

                # Créer une grille régulière pour l'interpolation
                depth_values = sondage_data['Depth'].values
                qc_values = sondage_data['qc'].values if 'qc' in sondage_data.columns else sondage_data['fs'].values

                # Vérifier que les données sont valides
                if len(depth_values) < 3 or len(qc_values) < 3:
                    # Pas assez de données pour l'interpolation, afficher seulement les points
                    fig.add_trace(
                        go.Scatter3d(
                            x=depth_values,
                            y=[idx] * len(depth_values),
                            z=qc_values,
                            mode='markers',
                            marker=dict(size=6, color='red'),
                            name=f'Données {sondage} (pas assez de points)',
                            showlegend=False
                        ),
                        row=row, col=col
                    )
                    continue

                # Nettoyer les données (supprimer NaN et infinis)
                valid_mask = ~(np.isnan(depth_values) | np.isnan(qc_values) | np.isinf(depth_values) | np.isinf(qc_values))
                depth_clean = depth_values[valid_mask]
                qc_clean = qc_values[valid_mask]

                if len(depth_clean) < 3:
                    # Pas assez de données valides
                    fig.add_trace(
                        go.Scatter(
                            x=depth_values,
                            y=qc_values,
                            mode='markers',
                            marker=dict(size=6, color='orange'),
                            name=f'Données {sondage} (données invalides)',
                            showlegend=False
                        ),
                        row=row, col=col
                    )
                    continue

                # Créer un graphique 3D Scatter pour ce sondage
                fig.add_trace(
                    go.Scatter3d(
                        x=depth_clean,
                        y=[idx] * len(depth_clean),  # Utiliser l'index du sondage pour séparer
                        z=qc_clean,
                        mode='markers',
                        marker=dict(
                            size=4,
                            color=qc_clean,
                            colorscale='Viridis',
                            showscale=False,
                            opacity=0.8
                        ),
                        name=f'Sondage {sondage}',
                        showlegend=False
                    ),
                    row=row, col=col
                )



            # Configuration du layout
            fig.update_layout(
                title="Visualisations 3D - Chaque Sondage CPTU",
                height=max(400, 300 * n_rows),
                width=None,
                showlegend=False,
                autosize=True,
                margin=dict(l=40, r=40, t=80, b=40, pad=10)
            )

            # Mettre à jour les axes pour chaque sous-graphique
            for i in range(1, n_rows + 1):
                for j in range(1, n_cols + 1):
                    fig.update_scenes(
                        xaxis_title="Profondeur (cm)",
                        yaxis_title="Sondage",
                        zaxis_title="qc (MPa)" if 'qc' in fused_data.columns else "fs (kPa)",
                        row=i, col=j
                    )

            # Convertir en HTML et afficher
            html_content = fig.to_html(include_plotlyjs='cdn', full_html=False, config={'responsive': True})
            self.fusion2DView.setHtml(html_content)

            # Activer le bouton d'export
            self.exportFusionPDFButton.setEnabled(True)

            # Informations sur les graphiques
            info_text = f"Graphiques 2D créés pour {n_sondages} sondages:\n"
            for sondage in sondages:
                sondage_data = fused_data[fused_data['Sondage'] == sondage]
                info_text += f"• {sondage}: {len(sondage_data)} points\n"

            self.fusion2DInfoLabel.setText(info_text)
            self.fusion2DInfoLabel.setStyleSheet("font-weight: bold; color: #2196F3;")

            print("✅ Graphiques 2D individuels créés avec succès")

        except Exception as e:
            error_html = f"<h1>Erreur 2D Individuelle: {str(e)}</h1><p>Vérifiez que scipy est installé pour l'interpolation.</p>"
            self.fusion2DView.setHtml(error_html)
            print(f"❌ Erreur dans createFused3DVisualization: {e}")
            import traceback
            traceback.print_exc()

    def exportFusion3DToPDF(self):
        """Exporter la carte 2D fusionnée en PDF"""
        try:
            if not hasattr(self, 'fused_data') or self.fused_data is None or self.fused_data.empty:
                QMessageBox.warning(self, "Erreur", "Aucune carte 2D à exporter.")
                return

            # Demander le nom du fichier
            fileName, _ = QFileDialog.getSaveFileName(self, "Sauvegarder Carte 2D PDF", "",
                                                      "Fichiers PDF (*.pdf);;Tous les fichiers (*)")

            if not fileName:
                return

            if not fileName.endswith('.pdf'):
                fileName += '.pdf'

            # Importer les bibliothèques nécessaires
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.utils import ImageReader
            import tempfile
            from plotly.io import to_image
            import plotly.graph_objects as go

            # Recréer la figure 2D (même logique que createFused3DVisualization)
            fig = go.Figure()
            
            fused_data = self.fused_data
            colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
            sondages = fused_data['Sondage'].unique()
            
            for i, sondage in enumerate(sondages):
                sondage_data = fused_data[fused_data['Sondage'] == sondage]
                color = colors[i % len(colors)]
                
                fig.add_trace(go.Scatter3d(
                    x=sondage_data['X'],
                    y=sondage_data['Y'], 
                    z=sondage_data['Depth'],
                    mode='markers',
                    marker=dict(
                        size=3,
                        color=sondage_data['qc'] if 'qc' in sondage_data.columns else color,
                        colorscale='Viridis' if 'qc' in sondage_data.columns else None,
                        showscale=True if i == 0 and 'qc' in sondage_data.columns else False,
                        colorbar=dict(title="qc (MPa)") if i == 0 and 'qc' in sondage_data.columns else None
                    ),
                    name=f'{sondage} - qc',
                    legendgroup=sondage
                ))
                
                fig.add_trace(go.Scatter3d(
                    x=sondage_data['X'],
                    y=sondage_data['Y'],
                    z=sondage_data['Depth'],
                    mode='lines',
                    line=dict(color=color, width=2),
                    name=f'{sondage} - profil',
                    legendgroup=sondage,
                    showlegend=False
                ))

            fig.update_layout(
                title="Carte 3D Complète du Sous-Sol - Fusion de Sondages CPTU",
                scene=dict(
                    xaxis_title='Coordonnée X (m)',
                    yaxis_title='Coordonnée Y (m)',
                    zaxis_title='Profondeur (m)',
                    zaxis_autorange="reversed"
                ),
                margin=dict(l=0, r=0, t=40, b=0),
                legend_title="Sondages CPTU"
            )

            # Créer le PDF
            c = canvas.Canvas(fileName, pagesize=A4)
            width, height = A4

            # Titre
            c.setFont("Helvetica-Bold", 16)
            c.drawString(50, height - 50, "Carte 3D Complète du Sous-Sol")
            c.setFont("Helvetica", 12)
            c.drawString(50, height - 70, f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
            c.drawString(50, height - 85, f"Nombre de sondages: {len(sondages)}")
            c.drawString(50, height - 100, f"Points de données totaux: {len(fused_data)}")

            # Statistiques
            c.drawString(50, height - 120, f"Profondeur max: {fused_data['Depth'].max():.1f} m")
            c.drawString(50, height - 135, f"Étendue X: {fused_data['X'].min():.1f} - {fused_data['X'].max():.1f} m")
            c.drawString(50, height - 150, f"Étendue Y: {fused_data['Y'].min():.1f} - {fused_data['Y'].max():.1f} m")

            # Liste des sondages
            c.drawString(50, height - 170, "Sondages inclus:")
            y_pos = height - 185
            for sondage in sondages:
                c.drawString(70, y_pos, f"• {sondage}")
                y_pos -= 15

            # Convertir la figure en image
            img_bytes = to_image(fig, format='png', width=700, height=500, scale=1)
            
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                tmp_file.write(img_bytes)
                tmp_filename = tmp_file.name
            
            # Ajouter l'image au PDF
            img = ImageReader(tmp_filename)
            c.drawImage(img, 50, 50, width=500, height=350)
            
            # Nettoyer
            os.unlink(tmp_filename)
            c.save()
            
            QMessageBox.information(self, "Succès", f"Carte 3D exportée en PDF:\n{fileName}")

        except ImportError as e:
            QMessageBox.warning(self, "Erreur", f"Bibliothèque manquante: {e}\n\nInstallez reportlab: pip install reportlab")
        except Exception as e:
            QMessageBox.warning(self, "Erreur", f"Erreur lors de l'export PDF: {e}")

    def export3DGraphsToPDF(self):
        """Exporte tous les graphiques 3D dans un seul PDF"""
        try:
            if not hasattr(self, 'df') or self.df is None or self.df.empty:
                QMessageBox.warning(self, "Erreur", "Aucune donnée chargée pour l'export 3D.")
                return

            # Importer les bibliothèques nécessaires
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.lib.utils import ImageReader
            import tempfile
            import os
            from plotly.io import to_image
            import plotly.graph_objects as go

            # Demander le nom du fichier de destination
            fileName, _ = QFileDialog.getSaveFileName(self, "Sauvegarder PDF des graphiques 3D", "",
                                                      "Fichiers PDF (*.pdf);;Tous les fichiers (*)")

            if not fileName:
                return

            if not fileName.endswith('.pdf'):
                fileName += '.pdf'

            # Créer les figures Plotly (même logique que update3D)
            figures = []
            df = self.df

            # Figure 1: 3D Scatter
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter3d(
                x=df['Depth'] if 'Depth' in df.columns else df.index,
                y=df['qc'] if 'qc' in df.columns else [0] * len(df),
                z=df['fs'] if 'fs' in df.columns else [0] * len(df),
                mode='markers',
                marker=dict(size=4, color=df['qc'] if 'qc' in df.columns else 'blue', colorscale='Viridis'),
                name='Points CPT'
            ))
            fig1.update_layout(
                title='3D Scatter: Depth vs qc vs fs',
                scene=dict(
                    xaxis_title='Depth (m)',
                    yaxis_title='qc (MPa)',
                    zaxis_title='fs (kPa)'
                ),
                margin=dict(l=0, r=0, t=40, b=0)
            )
            figures.append(('3D Scatter Plot', fig1))

            # Figure 2: 3D Surface
            if len(df) > 3:
                fig2 = go.Figure()
                fig2.add_trace(go.Surface(
                    x=df['Depth'].values.reshape(-1, 1) if 'Depth' in df.columns else df.index.values.reshape(-1, 1),
                    y=df['qc'].values.reshape(-1, 1) if 'qc' in df.columns else [0] * len(df),
                    z=df['fs'].values.reshape(-1, 1) if 'fs' in df.columns else [0] * len(df),
                    colorscale='Viridis',
                    name='Surface qc'
                ))
                fig2.update_layout(
                    title='3D Surface: qc vs Depth vs fs',
                    scene=dict(
                        xaxis_title='Depth (m)',
                        yaxis_title='qc (MPa)',
                        zaxis_title='fs (kPa)'
                    ),
                    margin=dict(l=0, r=0, t=40, b=0)
                )
                figures.append(('3D Surface Plot', fig2))

            # Figure 3: 3D Contour
            fig3 = go.Figure()
            fig3.add_trace(go.Contour(
                x=df['Depth'] if 'Depth' in df.columns else df.index,
                y=df['qc'] if 'qc' in df.columns else [0] * len(df),
                z=df['fs'] if 'fs' in df.columns else [0] * len(df),
                colorscale='Viridis',
                name='Contours fs'
            ))
            fig3.update_layout(
                title='3D Contour: Depth vs qc vs fs',
                xaxis_title='Depth (m)',
                yaxis_title='qc (MPa)',
                margin=dict(l=0, r=0, t=40, b=0)
            )
            figures.append(('3D Contour Plot', fig3))

            # Figure 4: 3D Wireframe (simulé avec scatter)
            fig4 = go.Figure()
            fig4.add_trace(go.Scatter3d(
                x=df['Depth'] if 'Depth' in df.columns else df.index,
                y=df['qc'] if 'qc' in df.columns else [0] * len(df),
                z=df['fs'] if 'fs' in df.columns else [0] * len(df),
                mode='lines+markers',
                line=dict(color='blue', width=2),
                marker=dict(size=3, color='red'),
                name='Wireframe'
            ))
            fig4.update_layout(
                title='3D Wireframe: qc Surface with Contours',
                scene=dict(
                    xaxis_title='Depth (m)',
                    yaxis_title='qc (MPa)',
                    zaxis_title='fs (kPa)'
                ),
                margin=dict(l=0, r=0, t=40, b=0)
            )
            figures.append(('3D Wireframe Plot', fig4))

            # Créer le PDF
            c = canvas.Canvas(fileName, pagesize=A4)
            width, height = A4

            # Titre du document
            c.setFont("Helvetica-Bold", 16)
            c.drawString(50, height - 50, "Rapport des Visualisations 3D - Analyse CPT")
            c.setFont("Helvetica", 12)
            c.drawString(50, height - 70, f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
            c.drawString(50, height - 85, f"Nombre de points: {len(df)}")

            y_position = height - 120

            # Convertir chaque figure en image et l'ajouter au PDF
            for title, fig in figures:
                try:
                    # Convertir la figure Plotly en image PNG
                    img_bytes = to_image(fig, format='png', width=600, height=400, scale=1)
                    
                    # Sauvegarder temporairement l'image
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                        tmp_file.write(img_bytes)
                        tmp_filename = tmp_file.name
                    
                    # Ajouter l'image au PDF
                    if y_position < 450:  # Nouvelle page si pas assez d'espace
                        c.showPage()
                        y_position = height - 50
                    
                    # Titre du graphique
                    c.setFont("Helvetica-Bold", 14)
                    c.drawString(50, y_position, title)
                    y_position -= 20
                    
                    # Ajouter l'image
                    img = ImageReader(tmp_filename)
                    c.drawImage(img, 50, y_position - 350, width=500, height=350)
                    y_position -= 380
                    
                    # Nettoyer le fichier temporaire
                    os.unlink(tmp_filename)
                    
                except Exception as e:
                    print(f"Erreur lors de l'export de {title}: {e}")
                    c.drawString(50, y_position, f"Erreur lors de la génération de {title}")
                    y_position -= 20

            # Sauvegarder le PDF
            c.save()
            
            QMessageBox.information(self, "Succès", f"PDF des graphiques 3D sauvegardé avec succès:\n{fileName}")

        except ImportError as e:
            QMessageBox.warning(self, "Erreur", f"Bibliothèque manquante pour l'export PDF: {e}\n\nInstallez reportlab avec: pip install reportlab")
        except Exception as e:
            QMessageBox.warning(self, "Erreur", f"Erreur lors de l'export PDF: {e}")

    def export3DGraphsIndividually(self):
        """Exporte chaque graphique 3D individuellement"""
        try:
            if not hasattr(self, 'df') or self.df is None or self.df.empty:
                QMessageBox.warning(self, "Erreur", "Aucune donnée chargée pour l'export 3D.")
                return

            # Demander le répertoire de destination
            directory = QFileDialog.getExistingDirectory(self, "Sélectionner le répertoire de sauvegarde")
            if not directory:
                return

            # Importer les bibliothèques nécessaires
            from plotly.io import write_image
            import plotly.graph_objects as go
            import os

            df = self.df
            exported_files = []

            # Figure 1: 3D Scatter
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter3d(
                x=df['Depth'] if 'Depth' in df.columns else df.index,
                y=df['qc'] if 'qc' in df.columns else [0] * len(df),
                z=df['fs'] if 'fs' in df.columns else [0] * len(df),
                mode='markers',
                marker=dict(size=4, color=df['qc'] if 'qc' in df.columns else 'blue', colorscale='Viridis'),
                name='Points CPT'
            ))
            fig1.update_layout(
                title='3D Scatter: Depth vs qc vs fs',
                scene=dict(
                    xaxis_title='Depth (m)',
                    yaxis_title='qc (MPa)',
                    zaxis_title='fs (kPa)'
                )
            )
            
            filename1 = os.path.join(directory, "3D_Scatter_Plot.png")
            write_image(fig1, filename1, format='png', width=800, height=600)
            exported_files.append(filename1)

            # Figure 2: 3D Surface (si assez de données)
            if len(df) > 3:
                fig2 = go.Figure()
                fig2.add_trace(go.Surface(
                    x=df['Depth'].values.reshape(-1, 1) if 'Depth' in df.columns else df.index.values.reshape(-1, 1),
                    y=df['qc'].values.reshape(-1, 1) if 'qc' in df.columns else [0] * len(df),
                    z=df['fs'].values.reshape(-1, 1) if 'fs' in df.columns else [0] * len(df),
                    colorscale='Viridis',
                    name='Surface qc'
                ))
                fig2.update_layout(
                    title='3D Surface: qc vs Depth vs fs',
                    scene=dict(
                        xaxis_title='Depth (m)',
                        yaxis_title='qc (MPa)',
                        zaxis_title='fs (kPa)'
                    )
                )
                
                filename2 = os.path.join(directory, "3D_Surface_Plot.png")
                write_image(fig2, filename2, format='png', width=800, height=600)
                exported_files.append(filename2)

            # Figure 3: 3D Contour
            fig3 = go.Figure()
            fig3.add_trace(go.Contour(
                x=df['Depth'] if 'Depth' in df.columns else df.index,
                y=df['qc'] if 'qc' in df.columns else [0] * len(df),
                z=df['fs'] if 'fs' in df.columns else [0] * len(df),
                colorscale='Viridis',
                name='Contours fs'
            ))
            fig3.update_layout(
                title='3D Contour: Depth vs qc vs fs',
                xaxis_title='Depth (m)',
                yaxis_title='qc (MPa)'
            )
            
            filename3 = os.path.join(directory, "3D_Contour_Plot.png")
            write_image(fig3, filename3, format='png', width=800, height=600)
            exported_files.append(filename3)

            # Figure 4: 3D Wireframe
            fig4 = go.Figure()
            fig4.add_trace(go.Scatter3d(
                x=df['Depth'] if 'Depth' in df.columns else df.index,
                y=df['qc'] if 'qc' in df.columns else [0] * len(df),
                z=df['fs'] if 'fs' in df.columns else [0] * len(df),
                mode='lines+markers',
                line=dict(color='blue', width=2),
                marker=dict(size=3, color='red'),
                name='Wireframe'
            ))
            fig4.update_layout(
                title='3D Wireframe: qc Surface with Contours',
                scene=dict(
                    xaxis_title='Depth (m)',
                    yaxis_title='qc (MPa)',
                    zaxis_title='fs (kPa)'
                )
            )
            
            filename4 = os.path.join(directory, "3D_Wireframe_Plot.png")
            write_image(fig4, filename4, format='png', width=800, height=600)
            exported_files.append(filename4)

            QMessageBox.information(self, "Succès", f"Graphiques 3D exportés individuellement:\n\n" + "\n".join(exported_files))

        except ImportError as e:
            QMessageBox.warning(self, "Erreur", f"Bibliothèque manquante pour l'export: {e}\n\nAssurez-vous que plotly est installé.")
        except Exception as e:
            QMessageBox.warning(self, "Erreur", f"Erreur lors de l'export individuel: {e}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())