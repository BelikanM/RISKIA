# Script PowerShell pour lancer VS Code avec Python portable
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "   LANCEMENT VS CODE AVEC PYTHON PORTABLE" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

$vsCodePath = "C:\Users\Admin\Desktop\logiciel\KIBALISTONEGOD"
$pythonPath = "C:\Users\Admin\Desktop\logiciel\KIBALISTONEGOD\python311\python.exe"

# Vérification de VS Code
Write-Host "🔍 Vérification de VS Code..." -ForegroundColor Yellow
try {
    $codeVersion = & code --version 2>$null | Select-Object -First 1
    Write-Host "✅ VS Code détecté: $codeVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ VS Code n'est pas installé ou n'est pas dans le PATH" -ForegroundColor Red
    Write-Host "Veuillez installer VS Code et l'ajouter au PATH système" -ForegroundColor Yellow
    Read-Host "Appuyez sur Entrée pour quitter"
    exit 1
}

Write-Host ""
Write-Host "🚀 Lancement de VS Code avec le dossier KIBALISTONEGOD..." -ForegroundColor Green
Write-Host "Le Python portable sera automatiquement utilisé" -ForegroundColor Cyan
Write-Host ""

# Lancement de VS Code
& code $vsCodePath --new-window

Write-Host ""
Write-Host "✅ VS Code lancé avec la configuration Python portable" -ForegroundColor Green
Write-Host "Les erreurs Pylance devraient disparaître automatiquement" -ForegroundColor Cyan
Write-Host ""

Read-Host "Appuyez sur Entrée pour quitter"