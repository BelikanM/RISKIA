import re
import math
import sympy as sp

def calculate_expression(expression: str):
    """
    Évalue une expression mathématique de manière sécurisée.
    Supporte les opérations de base, fonctions trigonométriques, logarithmes, etc.
    """
    try:
        # Nettoyer l'expression
        expr = expression.strip()

        # Vérifier si c'est une expression mathématique (pas de lettres)
        if any(char.isalpha() for char in expr.replace(' ', '')):
            return f"Expression non mathématique: {expr[:50]}..."

        # Remplacer les mots par des symboles
        replacements = {
            'pi': 'math.pi',
            'e': 'math.e',
            'sqrt': 'math.sqrt',
            'sin': 'math.sin',
            'cos': 'math.cos',
            'tan': 'math.tan',
            'log': 'math.log',
            'ln': 'math.log',
            'exp': 'math.exp',
            '^': '**'
        }

        for word, replacement in replacements.items():
            expr = re.sub(r'\b' + re.escape(word) + r'\b', replacement, expr)

        # Évaluer l'expression de manière sécurisée
        allowed_names = {
            "math": math,
            "__builtins__": {}
        }

        result = eval(expr, allowed_names)

        # Formater le résultat
        if isinstance(result, float):
            if result.is_integer():
                return int(result)
            return round(result, 6)
        return result

    except Exception as e:
        return f"Erreur de calcul: {str(e)}"

def extract_math_from_text(text: str):
    """
    Extrait et résout les expressions mathématiques d'un texte.
    """
    # Patterns pour les expressions mathématiques
    patterns = [
        r'\d+(?:\.\d+)?\s*[\+\-\*\/\^]\s*\d+(?:\.\d+)?(?:\s*[\+\-\*\/\^]\s*\d+(?:\.\d+)?)*',  # Opérations de base
        r'sqrt\(\d+(?:\.\d+)?\)',  # Racine carrée
        r'\d+(?:\.\d+)?\s*\^\s*\d+(?:\.\d+)?',  # Puissance
    ]

    results = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            result = calculate_expression(match)
            if not isinstance(result, str) or not result.startswith("Erreur"):
                results.append(f"{match} = {result}")

    return results if results else None