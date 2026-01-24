from tools.conversation_history import search_conversation_history
import logging

logger = logging.getLogger(__name__)

def context_search(query: str) -> dict:
    """
    Recherche le contexte dans l'historique des conversations pour une phrase non précise.
    Détermine si la requête se rattache à un sujet existant ou s'il s'agit d'un nouveau sujet.

    Args:
        query (str): La phrase ou requête à analyser.

    Returns:
        dict: {
            "is_new_subject": bool,
            "context": str,  # Contexte pertinent si trouvé
            "relevant_messages": list,  # Messages pertinents
            "confidence": float  # Confiance dans la liaison (0-1)
        }
    """
    try:
        # Rechercher dans l'historique
        relevant_messages = search_conversation_history(query, limit=5)

        if not relevant_messages:
            return {
                "is_new_subject": True,
                "context": "",
                "relevant_messages": [],
                "confidence": 0.0
            }

        # Analyser la pertinence
        total_score = sum(msg.get("score", 0) for msg in relevant_messages)
        max_score = max((msg.get("score", 0) for msg in relevant_messages), default=0)

        # Seuil de confiance : si le score max > 10, considérer comme lié (plus lenient)
        confidence = min(max_score / 15.0, 1.0)  # Normaliser sur 15

        is_new_subject = confidence < 0.4  # Seuil plus bas

        # Construire le contexte
        context_parts = []
        for msg in relevant_messages[:3]:  # Limiter à 3 messages
            role = "Utilisateur" if msg["role"] == "user" else "Assistant"
            content = msg["content"][:200] + "..." if len(msg["content"]) > 200 else msg["content"]
            context_parts.append(f"{role}: {content}")

        context = "\n".join(context_parts) if context_parts else ""

        return {
            "is_new_subject": is_new_subject,
            "context": context,
            "relevant_messages": relevant_messages,
            "confidence": confidence
        }

    except Exception as e:
        logger.error(f"Erreur dans context_search: {e}")
        return {
            "is_new_subject": True,
            "context": "",
            "relevant_messages": [],
            "confidence": 0.0
        }