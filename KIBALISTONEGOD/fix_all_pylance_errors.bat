@echo off
REM Script principal pour résoudre TOUTES les erreurs Pylance
echo ========================================
echo 🚀 SOLUTION COMPLÈTE POUR PYLANCE
echo ========================================
echo.

echo Ce script va:
echo 1. Installer toutes les dépendances dans python311
echo 2. Nettoyer le cache Pylance/VS Code
echo 3. Configurer correctement l'environnement
echo.

pause

echo.
echo ========================================
echo 📦 ÉTAPE 1: INSTALLATION DES DÉPENDANCES
echo ========================================
echo.

call install_all_dependencies.bat

if errorlevel 1 (
    echo ❌ Échec de l'installation des dépendances
    pause
    exit /b 1
)

echo.
echo ========================================
echo 🧹 ÉTAPE 2: NETTOYAGE DU CACHE
echo ========================================
echo.

call clean_pylance_cache.bat

echo.
echo ========================================
echo ✅ SOLUTION TERMINÉE!
echo ========================================
echo.
echo 🎯 PROCHAINES ÉTAPES:
echo.
echo 1️⃣ Fermez COMPLETEMENT VS Code
echo 2️⃣ Attendez 10 secondes
echo 3️⃣ Redémarrez VS Code
echo 4️⃣ Ouvrez le workspace KIBALISTONEGOD
echo 5️⃣ Ouvrez Dust3r.py
echo.
echo ✅ TOUTES les erreurs Pylance devraient avoir disparu!
echo.
echo 📋 RAPPORTS CRÉÉS:
echo - installation_report.txt : Rapport d'installation
echo.
echo 🔧 SCRIPTS DISPONIBLES:
echo - install_all_dependencies.bat : Réinstaller les dépendances
echo - clean_pylance_cache.bat : Nettoyer le cache
echo - final_pylance_fix.bat : Correction rapide
echo.

echo 🎉 PROFITEZ DE VOTRE CODAGE SANS ERREURS!
echo.

pause