# Script PowerShell pour lancer Dust3r
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "   LANCEMENT DUST3R - PHOTOGRAMMETRIE IA" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Chemin vers le Python portable
$pythonPath = "C:\Users\Admin\Desktop\logiciel\KIBALISTONEGOD\python311\python.exe"
$scriptPath = "C:\Users\Admin\Desktop\logiciel\KIBALISTONEGOD\Dust3r.py"
$workingDir = "C:\Users\Admin\Desktop\logiciel\KIBALISTONEGOD"

# Vérification des dépendances
Write-Host "🔍 Vérification des dépendances..." -ForegroundColor Yellow
& $pythonPath -c "import streamlit, torch, PIL, numpy, plotly, open3d, transformers, pynvml, faiss, pandas, sklearn, psutil; print('✓ Toutes les dépendances sont présentes')"

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Erreur: Dépendances manquantes" -ForegroundColor Red
    Read-Host "Appuyez sur Entrée pour quitter"
    exit 1
}

Write-Host ""
Write-Host "🌐 Détection de l'adresse IP locale..." -ForegroundColor Yellow

# Détection de l'adresse IP
$ipAddress = (Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.InterfaceAlias -notlike "*Loopback*" -and $_.IPAddress -notlike "169.*" } | Select-Object -First 1).IPAddress

if ($ipAddress) {
    Write-Host "Adresse IP détectée: $ipAddress" -ForegroundColor Green
    Write-Host ""
    Write-Host "🚀 Lancement de l'application Streamlit..." -ForegroundColor Green
    Write-Host "L'application sera accessible sur:" -ForegroundColor Green
    Write-Host "  - http://localhost:8501" -ForegroundColor Cyan
    Write-Host "  - http://$($ipAddress):8501" -ForegroundColor Cyan
    Write-Host ""
} else {
    Write-Host "Adresse IP non détectée, utilisation de localhost uniquement" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "🚀 Lancement de l'application Streamlit..." -ForegroundColor Green
    Write-Host "L'application sera accessible sur http://localhost:8501" -ForegroundColor Cyan
    Write-Host ""
}

# Lancement de l'application
& $pythonPath -m streamlit run $scriptPath --server.port 8501 --server.address 0.0.0.0

Read-Host "Appuyez sur Entrée pour quitter"