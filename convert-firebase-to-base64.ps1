# Firebase Credentials to Base64 Converter for Render Deployment
# Run this script to convert your Firebase Admin SDK JSON file to base64

$firebasePath = "C:\Users\AIT 33\Documents\Secrets\firebase-admin-sdk.json"

if (-Not (Test-Path $firebasePath)) {
    Write-Host "ERROR: Firebase JSON file not found at: $firebasePath" -ForegroundColor Red
    Write-Host "Please update the path in this script or specify the correct path." -ForegroundColor Yellow
    $customPath = Read-Host "Enter the path to your Firebase JSON file (or press Enter to exit)"
    if ($customPath) {
        $firebasePath = $customPath
    } else {
        exit 1
    }
}

Write-Host "Reading Firebase credentials from: $firebasePath" -ForegroundColor Cyan

try {
    $fileContent = Get-Content $firebasePath -Raw
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($fileContent)
    $base64 = [Convert]::ToBase64String($bytes)
    
    # Copy to clipboard
    $base64 | Set-Clipboard
    
    Write-Host ""
    Write-Host "SUCCESS!" -ForegroundColor Green
    Write-Host "The base64-encoded Firebase credentials have been copied to your clipboard." -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Yellow
    Write-Host "1. Go to Render Dashboard - Your Service - Environment" -ForegroundColor White
    Write-Host "2. Add environment variable:" -ForegroundColor White
    Write-Host "   Key: FIREBASE_SERVICE_ACCOUNT_BASE64" -ForegroundColor Cyan
    Write-Host "   Value: Paste from clipboard (Ctrl+V)" -ForegroundColor Cyan
    Write-Host "3. Also add FIREBASE_PROJECT_ID with your Firebase project ID" -ForegroundColor White
    Write-Host ""
    Write-Host "The base64 string is also shown below (first 100 characters):" -ForegroundColor Yellow
    Write-Host $base64.Substring(0, [Math]::Min(100, $base64.Length)) -ForegroundColor Gray
    Write-Host "..." -ForegroundColor Gray
    
} catch {
    Write-Host ""
    Write-Host "ERROR: Failed to convert Firebase credentials" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}
