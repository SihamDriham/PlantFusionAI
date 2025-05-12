import streamlit as st
import matplotlib.pyplot as plt
from config.settings import FRIENDLY_CLASS_NAMES, UI_COLORS

def info_box(text, icon="ℹ️"):
    """
    Crée une box d'information stylisée
    
    Args:
        text (str): Texte à afficher
        icon (str, optional): Icône à utiliser. Defaults to "ℹ️".
    """
    st.markdown(f"""
        <div class="info-box">
            <p>{icon} {text}</p>
        </div>
    """, unsafe_allow_html=True)

def get_friendly_class_name(class_name):
    """
    Convertit le nom technique de la classe en nom convivial pour l'affichage
    
    Args:
        class_name (str): Nom technique de la classe
    
    Returns:
        str: Nom convivial pour l'affichage
    """
    return FRIENDLY_CLASS_NAMES.get(class_name, class_name)

def display_result(result, confidence):
    """
    Affiche le résultat de la classification avec sa confiance
    
    Args:
        result (str): Résultat de la classification
        confidence (float): Score de confiance
    """
    # Obtenir le nom convivial pour l'affichage
    friendly_result = get_friendly_class_name(result)
    
    # Définir le style de la boîte en fonction du résultat
    if result == "Healthy":
        box_class = "healthy-box"
        icon = "✅"
    else:
        box_class = "diseased-box"
        icon = "❌"
    
    st.markdown(f"""
        <div class="status-box {box_class}">
            <h2 style="margin-top: 0;">{icon} État de la plante: {friendly_result}</h2>
            <p style="font-size: 1.2rem;">Confiance: <b>{confidence:.2%}</b></p>
        </div>
    """, unsafe_allow_html=True)

def display_fusion_visualization(fusion_type):
    """Affiche une visualisation améliorée de la fusion pour les modèles multimodaux"""
    # Création d'un graphique pour illustrer le type de fusion
    fig, ax = plt.subplots(figsize=(7, 4))
    
    # Coordonnées pour les différents éléments
    if fusion_type == "Early Fusion":
        # Visualisation de Early Fusion avec schéma
        ax.add_patch(plt.Rectangle((0.1, 0.6), 0.2, 0.2, color='lightblue', alpha=0.7))
        ax.add_patch(plt.Rectangle((0.1, 0.3), 0.2, 0.2, color='lightgreen', alpha=0.7))
        
        ax.add_patch(plt.Rectangle((0.4, 0.45), 0.2, 0.2, color='orange', alpha=0.7))
        
        ax.add_patch(plt.Rectangle((0.7, 0.45), 0.2, 0.2, color='red', alpha=0.7))
        
        # Flèches
        ax.arrow(0.3, 0.7, 0.08, -0.15, head_width=0.03, head_length=0.03, fc='black', ec='black')
        ax.arrow(0.3, 0.4, 0.08, 0.15, head_width=0.03, head_length=0.03, fc='black', ec='black')
        ax.arrow(0.6, 0.55, 0.08, 0, head_width=0.03, head_length=0.03, fc='black', ec='black')
        
        # Texte
        ax.text(0.2, 0.7, "RGB", ha='center', va='center', fontsize=10)
        ax.text(0.2, 0.4, "MS", ha='center', va='center', fontsize=10)
        ax.text(0.5, 0.55, "Fusion", ha='center', va='center', fontsize=10)
        ax.text(0.8, 0.55, "Décision", ha='center', va='center', fontsize=10)
        
        ax.text(0.5, 0.85, "Early Fusion", ha='center', va='center', fontsize=14, fontweight='bold')
        ax.text(0.5, 0.15, "Fusion des données en entrée", ha='center', va='center', fontsize=12)
    
    elif fusion_type == "Hybrid Fusion":
        # Visualisation de Hybrid Fusion avec schéma
        ax.add_patch(plt.Rectangle((0.1, 0.7), 0.15, 0.15, color='lightblue', alpha=0.7))
        ax.add_patch(plt.Rectangle((0.1, 0.2), 0.15, 0.15, color='lightgreen', alpha=0.7))
        
        ax.add_patch(plt.Rectangle((0.35, 0.7), 0.15, 0.15, color='skyblue', alpha=0.7))
        ax.add_patch(plt.Rectangle((0.35, 0.2), 0.15, 0.15, color='darkseagreen', alpha=0.7))
        
        ax.add_patch(plt.Rectangle((0.6, 0.45), 0.15, 0.15, color='orange', alpha=0.7))
        
        ax.add_patch(plt.Rectangle((0.85, 0.45), 0.1, 0.15, color='red', alpha=0.7))
        
        # Flèches
        ax.arrow(0.25, 0.775, 0.08, 0, head_width=0.03, head_length=0.03, fc='black', ec='black')
        ax.arrow(0.25, 0.275, 0.08, 0, head_width=0.03, head_length=0.03, fc='black', ec='black')
        ax.arrow(0.5, 0.775, 0.08, -0.2, head_width=0.03, head_length=0.03, fc='black', ec='black')
        ax.arrow(0.5, 0.275, 0.08, 0.2, head_width=0.03, head_length=0.03, fc='black', ec='black')
        ax.arrow(0.75, 0.525, 0.08, 0, head_width=0.03, head_length=0.03, fc='black', ec='black')
        
        # Texte
        ax.text(0.175, 0.775, "RGB", ha='center', va='center', fontsize=8)
        ax.text(0.175, 0.275, "MS", ha='center', va='center', fontsize=8)
        ax.text(0.425, 0.775, "Conv", ha='center', va='center', fontsize=8)
        ax.text(0.425, 0.275, "Conv", ha='center', va='center', fontsize=8)
        ax.text(0.675, 0.525, "Fusion", ha='center', va='center', fontsize=8)
        ax.text(0.9, 0.525, "Déc", ha='center', va='center', fontsize=8)
        
        ax.text(0.5, 0.9, "Hybrid Fusion", ha='center', va='center', fontsize=14, fontweight='bold')
        ax.text(0.5, 0.1, "Fusion à plusieurs niveaux du réseau", ha='center', va='center', fontsize=12)
    
    else:  # "Late Fusion"
        # Visualisation de Late Fusion avec schéma
        ax.add_patch(plt.Rectangle((0.1, 0.7), 0.15, 0.15, color='lightblue', alpha=0.7))
        ax.add_patch(plt.Rectangle((0.1, 0.2), 0.15, 0.15, color='lightgreen', alpha=0.7))
        
        ax.add_patch(plt.Rectangle((0.35, 0.7), 0.15, 0.15, color='skyblue', alpha=0.7))
        ax.add_patch(plt.Rectangle((0.35, 0.2), 0.15, 0.15, color='darkseagreen', alpha=0.7))
        
        ax.add_patch(plt.Rectangle((0.6, 0.7), 0.15, 0.15, color='royalblue', alpha=0.7))
        ax.add_patch(plt.Rectangle((0.6, 0.2), 0.15, 0.15, color='green', alpha=0.7))
        
        ax.add_patch(plt.Rectangle((0.85, 0.45), 0.1, 0.15, color='red', alpha=0.7))
        
        # Flèches
        ax.arrow(0.25, 0.775, 0.08, 0, head_width=0.03, head_length=0.03, fc='black', ec='black')
        ax.arrow(0.25, 0.275, 0.08, 0, head_width=0.03, head_length=0.03, fc='black', ec='black')
        ax.arrow(0.5, 0.775, 0.08, 0, head_width=0.03, head_length=0.03, fc='black', ec='black')
        ax.arrow(0.5, 0.275, 0.08, 0, head_width=0.03, head_length=0.03, fc='black', ec='black')
        ax.arrow(0.75, 0.775, 0.08, -0.2, head_width=0.03, head_length=0.03, fc='black', ec='black')
        ax.arrow(0.75, 0.275, 0.08, 0.2, head_width=0.03, head_length=0.03, fc='black', ec='black')
        
        # Texte
        ax.text(0.175, 0.775, "RGB", ha='center', va='center', fontsize=8)
        ax.text(0.175, 0.275, "MS", ha='center', va='center', fontsize=8)
        ax.text(0.425, 0.775, "Conv", ha='center', va='center', fontsize=8)
        ax.text(0.425, 0.275, "Conv", ha='center', va='center', fontsize=8)
        ax.text(0.675, 0.775, "Class", ha='center', va='center', fontsize=8)
        ax.text(0.675, 0.275, "Class", ha='center', va='center', fontsize=8)
        ax.text(0.9, 0.525, "Fusion", ha='center', va='center', fontsize=8)
        
        ax.text(0.5, 0.9, "Late Fusion", ha='center', va='center', fontsize=14, fontweight='bold')
        ax.text(0.5, 0.1, "Fusion des résultats des différents modèles", ha='center', va='center', fontsize=12)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    st.pyplot(fig)

    """
    Affiche une visualisation améliorée de la fusion pour les modèles multimodaux
    
    Args:
        fusion_type (str): Type de fusion ('Early Fusion', 'Hybrid Fusion', 'Late Fusion')
    """
    # Création d'un graphique pour illustrer le type de fusion
    fig, ax = plt.subplots(figsize=(7, 4))
    
    # Coordonnées pour les différents éléments
    if fusion_type == "Early Fusion":
        # Visualisation de Early Fusion avec schéma
        ax.add_patch(plt.Rectangle((0.1, 0.6), 0.2, 0.2, color='lightblue', alpha=0.7))
        ax.add_patch(plt.Rectangle((0.1, 0.3), 0.2, 0.2, color='lightgreen', alpha=0.7))
        
        ax.add_patch(plt.Rectangle((0.4, 0.45), 0.2, 0.2, color='orange', alpha=0.7))
        
        ax.add_patch(plt.Rectangle((0.7, 0.45), 0.2, 0.2, color='red', alpha=0.7))
        
        # Flèches
        ax.arrow(0.3, 0.7, 0.08, -0.15, head_width=0.03, head_length=0.03, fc='black', ec='black')
        ax.arrow(0.3, 0.4, 0.08, 0.15, head_width=0.03, head_length=0.03, fc='black', ec='black')
        ax.arrow(0.6, 0.55, 0.08, 0, head_width=0.03, head_length=0.03, fc='black', ec='black')
        
        # Texte
        ax.text(0.2, 0.7, "RGB", ha='center', va='center', fontsize=10)
        ax.text(0.2, 0.4, "MS", ha='center', va='center', fontsize=10)
        ax.text(0.5, 0.55, "Fusion", ha='center', va='center', fontsize=10)
        ax.text(0.8, 0.55, "Décision", ha='center', va='center', fontsize=10)
        
        ax.text(0.5, 0.85, "Early Fusion", ha='center', va='center', fontsize=14, fontweight='bold')
        ax.text(0.5, 0.15, "Fusion des données en entrée", ha='center', va='center', fontsize=12)
    
    elif fusion_type == "Hybrid Fusion":
        # Visualisation de Hybrid Fusion avec schéma
        ax.add_patch(plt.Rectangle((0.1, 0.7), 0.15, 0.15, color='lightblue', alpha=0.7))
        ax.add_patch(plt.Rectangle((0.1, 0.2), 0.15, 0.15, color='lightgreen', alpha=0.7))
        
        ax.add_patch(plt.Rectangle((0.35, 0.7), 0.15, 0.15, color='skyblue', alpha=0.7))
        ax.add_patch(plt.Rectangle((0.35, 0.2), 0.15, 0.15, color='darkseagreen', alpha=0.7))
        
        ax.add_patch(plt.Rectangle((0.6, 0.45), 0.15, 0.15, color='orange', alpha=0.7))
        
        ax.add_patch(plt.Rectangle((0.85, 0.45), 0.1, 0.15, color='red', alpha=0.7))
        
        # Flèches
        ax.arrow(0.25, 0.775, 0.08, 0, head_width=0.03, head_length=0.03, fc='black', ec='black')
        ax.arrow(0.25, 0.275, 0.08, 0, head_width=0.03, head_length=0.03, fc='black', ec='black')
        ax.arrow(0.5, 0.775, 0.08, -0.2, head_width=0.03, head_length=0.03, fc='black', ec='black')
        ax.arrow(0.5, 0.275, 0.08, 0.2, head_width=0.03, head_length=0.03, fc='black', ec='black')
        ax.arrow(0.75, 0.525, 0.08, 0, head_width=0.03, head_length=0.03, fc='black', ec='black')
        
        # Texte
        ax.text(0.175, 0.775, "RGB", ha='center', va='center', fontsize=8)
        ax.text(0.175, 0.275, "MS", ha='center', va='center', fontsize=8)
        ax.text(0.425, 0.775, "Conv", ha='center', va='center', fontsize=8)
        ax.text(0.425, 0.275, "Conv", ha='center', va='center', fontsize=8)
        ax.text(0.675, 0.525, "Fusion", ha='center', va='center', fontsize=8)
        ax.text(0.9, 0.525, "Déc", ha='center', va='center', fontsize=8)
        
        ax.text(0.5, 0.9, "Hybrid Fusion", ha='center', va='center', fontsize=14, fontweight='bold')
        ax.text(0.5, 0.1, "Fusion à plusieurs niveaux du réseau", ha='center', va='center', fontsize=12)
    
    else:  # "Late Fusion"
        # Visualisation de Late Fusion avec schéma
        ax.add_patch(plt.Rectangle((0.1, 0.7), 0.15, 0.15, color='lightblue', alpha=0.7))
        ax.add_patch(plt.Rectangle((0.1, 0.2), 0.15, 0.15, color='lightgreen', alpha=0.7))
        
        ax.add_patch(plt.Rectangle((0.35, 0.7), 0.15, 0.15, color='skyblue', alpha=0.7))
        ax.add_patch(plt.Rectangle((0.35, 0.2), 0.15, 0.15, color='darkseagreen', alpha=0.7))
        
        ax.add_patch(plt.Rectangle((0.6, 0.7), 0.15, 0.15, color='royalblue', alpha=0.7))
        ax.add_patch(plt.Rectangle((0.6, 0.2), 0.15, 0.15, color='green', alpha=0.7))
        
        ax.add_patch(plt.Rectangle((0.85, 0.45), 0.1, 0.15, color='red', alpha=0.7))
        
        # Flèches
        ax.arrow(0.25, 0.775, 0.08, 0, head_width=0.03, head_length=0.03, fc='black', ec='black')
        ax.arrow(0.25, 0.275, 0.08, 0, head_width=0.03, head_length=0.03, fc='black', ec='black')
        ax.arrow(0.5, 0.775, 0.08, 0, head_width=0.03, head_length=0.03, fc='black', ec='black')
        ax.arrow(0.5, 0.275, 0.08, 0, head_width=0.03, head_length=0.03, fc='black', ec='black')
        ax.arrow(0.75, 0.775, 0.08, -0.2, head_width=0.03, head_length=0.03, fc='black', ec='black')
        ax.arrow(0.75, 0.275, 0.08, 0.2)