import streamlit as st
from PIL import Image


def main():
    # Configuration de la page
    st.set_page_config(
        page_title="Présentation de l'Application",
        page_icon="🤖",
        layout="centered",
        initial_sidebar_state="expanded"
    )

    # Chargement des images et logos
    # Remplacez par vos propres images si besoin
    logo = Image.open("image/mltools-high-resolution-logo-transparent.png")  # Remplacez par le chemin de votre logo
    background_image = Image.open("image/background.png")  # Remplacez par une image de fond appropriée

    # En-tête principal avec logo
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(logo, width=350,)
        st.markdown(
            "<h1 style='text-align: center; color: #1EBBD7;'>Bienvenue dans l'Application d'Extraction et de Traitement de Données</h1>",
            unsafe_allow_html=True
        )

    # Sous-titre accrocheur
    st.markdown(
        """
        <h3 style='text-align: center; color: #555555;'>Une solution puissante pour l'extraction, le traitement, et l'analyse de données au sein d'une interface intuitive</h3>
        """,
        unsafe_allow_html=True
    )

    # Section sur les fonctionnalités principales
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center;">
            <h2>Fonctionnalités Principales 🚀</h2>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.write(
        """
        - **Extraction de Données** : Extraire facilement des données depuis un fichier local, un bucket S3, ou une base de données PostgreSQL.
        - **Prétraitement et Fusion** : Nettoyer, prétraiter, et fusionner plusieurs fichiers en un seul jeu de données consolidé.
        - **Entraînement de Modèles** : Entraînez rapidement un modèle XGBoost sur les données extraites pour une analyse prédictive avancée.
        - **Visualisation et Analyse** : Obtenez des métriques et visualisez les résultats de vos modèles.
        """
    )

    # Images ou icônes pour chaque fonctionnalité (ajoutez les vôtres)
    col1, col2, col3 = st.columns(3)
    col1.image("image/machine-learning.png", width=200)  # Remplacez avec une icône pour l'extraction
    col2.image("image/extraction.png", width=200)  # Remplacez avec une icône pour le prétraitement
    col3.image("image/computer.png", width=200)  # Remplacez avec une icône pour l'entraînement

    # Section des avantages
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center;">
            <h2>Pourquoi Utiliser Cette Application ? 💡</h2>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.write(
        """
        - **Gain de Temps** : Automatisez l'extraction et le traitement des données en quelques clics.
        - **Facilité d'Utilisation** : Interface intuitive qui vous guide à travers chaque étape du processus.
        - **Personnalisation** : Adaptez les paramètres d'extraction et d'entraînement en fonction de vos besoins.
        - **Sécurité des Données** : Toutes les données restent dans votre environnement sécurisé.
        """
    )

    # Appel à l'action
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center;">
            <h2>Prêt à Commencer ?</h2>
            <p>Explorez les fonctionnalités dès maintenant et transformez vos données en informations exploitables.</p>
            <a href="#app" target="_self" style="text-decoration: none;">
                <button style="background-color: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 5px; font-size: 18px; cursor: pointer;">
                    Accéder à l'Application
                </button>
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Footer ou note supplémentaire
    st.markdown("---")
    st.markdown(
        """
        <footer style="text-align: center;">
            <p>Développé avec ❤️ par votre équipe. Toutes les données sont sécurisées et protégées.</p>
        </footer>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
