# Lanceur DUSt3R Portable - Reconstruction 3D Ultra-Réaliste
# Détection automatique des URLs et adresses IP

Write-Host "🚀 Lancement de DUSt3R - Reconstruction 3D Photogrammétrique Ultra-Réaliste" -ForegroundColor Green
Write-Host "=================================================================" -ForegroundColor Yellow

# Détection de l'adresse IP locale
$ipAddress = (Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.InterfaceAlias -notlike "*Loopback*" -and $_.IPAddress -like "192.*" } | Select-Object -First 1).IPAddress

if (-not $ipAddress) {
    $ipAddress = (Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.InterfaceAlias -notlike "*Loopback*" } | Select-Object -First 1).IPAddress
}

Write-Host "📡 Adresse IP détectée : $ipAddress" -ForegroundColor Cyan
Write-Host "🌐 URLs d'accès :" -ForegroundColor Cyan
Write-Host "   Local : http://localhost:8501" -ForegroundColor White
Write-Host "   Réseau : http://$($ipAddress):8501" -ForegroundColor White
Write-Host "" -ForegroundColor Cyan

# Chemin vers Python portable
$pythonPath = "$PSScriptRoot\python311\python.exe"
$scriptPath = "$PSScriptRoot\Dust3r.py"

Write-Host "🐍 Utilisation de Python portable : $pythonPath" -ForegroundColor Magenta
Write-Host "📄 Script : $scriptPath" -ForegroundColor Magenta
Write-Host "" -ForegroundColor Magenta

Write-Host "⏳ Démarrage de l'application Streamlit..." -ForegroundColor Yellow
Write-Host "=================================================================" -ForegroundColor Yellow

# Lancement de Streamlit
& $pythonPath -m streamlit run $scriptPath

Write-Host ""
Write-Host "✅ Application arrêtée." -ForegroundColor Green