# Deployment Guide for Render

## Prerequisites
1. A Render account (sign up at https://render.com)
2. Your Firebase Admin SDK JSON file
3. Git repository (GitHub, GitLab, or Bitbucket)

## Step 1: Prepare Firebase Credentials

Since Render doesn't have direct access to your local Firebase JSON file, you need to convert it to base64 and add it as an environment variable.

### On Windows PowerShell:
```powershell
$fileContent = Get-Content "C:\Users\AIT 33\Documents\Secrets\firebase-admin-sdk.json" -Raw
$bytes = [System.Text.Encoding]::UTF8.GetBytes($fileContent)
$base64 = [Convert]::ToBase64String($bytes)
$base64 | Set-Clipboard
Write-Host "Base64 encoded Firebase credentials copied to clipboard!"
```

This will copy the base64-encoded Firebase credentials to your clipboard.

## Step 2: Push to Git Repository

If you haven't already, initialize a git repository and push to GitHub/GitLab:

```bash
cd "c:\Users\AIT 33\Desktop\waste2"
git init
git add .
git commit -m "Initial commit for Render deployment"
git branch -M main
git remote add origin <YOUR_REPOSITORY_URL>
git push -u origin main
```

## Step 3: Deploy on Render

### Option A: Using render.yaml (Recommended)

1. Go to https://dashboard.render.com
2. Click "New +" → "Blueprint"
3. Connect your GitHub/GitLab repository
4. Render will automatically detect the `render.yaml` file
5. Click "Apply" to create the service

### Option B: Manual Web Service Creation

1. Go to https://dashboard.render.com
2. Click "New +" → "Web Service"
3. Connect your repository
4. Configure the service:
   - **Name**: waste-volume-estimator
   - **Region**: Choose closest to you (e.g., Oregon)
   - **Branch**: main
   - **Root Directory**: (leave empty)
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn --bind 0.0.0.0:$PORT web.app:app`
   - **Plan**: Free

## Step 4: Configure Environment Variables

After creating the service, add these environment variables in Render dashboard:

1. Go to your service → "Environment" tab
2. Add the following environment variables:

| Key | Value |
|-----|-------|
| `FLASK_ENV` | `production` |
| `FIREBASE_PROJECT_ID` | Your Firebase project ID (find in Firebase Console) |
| `FIREBASE_SERVICE_ACCOUNT_BASE64` | Paste the base64 string from Step 1 |
| `PORT` | `10000` (Render provides this automatically) |

**Important**: Make sure to use `FIREBASE_SERVICE_ACCOUNT_BASE64` and NOT `FIREBASE_CREDENTIALS_FILE` since file paths don't work on Render.

## Step 5: Deploy

1. Click "Manual Deploy" → "Deploy latest commit"
2. Wait for the build to complete (5-10 minutes)
3. Once deployed, your app will be available at: `https://waste-volume-estimator.onrender.com`

## Step 6: Initialize Firestore Database

If you haven't already set up Firestore:

1. Go to Firebase Console (https://console.firebase.google.com)
2. Select your project
3. Go to "Firestore Database"
4. Click "Create database"
5. Choose production mode or test mode
6. Select a location
7. The collection `utensil_sessions` will be created automatically when the first session is saved

## Troubleshooting

### Build Fails
- Check logs for missing dependencies
- Ensure `opencv-python-headless` is used (not `opencv-python`)
- Verify Python version compatibility

### Firebase Connection Errors
- Verify `FIREBASE_SERVICE_ACCOUNT_BASE64` is correctly encoded
- Check that your Firebase project ID is correct
- Ensure Firestore is enabled in Firebase Console
- Check that the service account has appropriate permissions

### Application Errors
- Check the logs in Render dashboard
- Ensure all environment variables are set correctly
- Test locally first with production settings

## Local Testing

Test your deployment configuration locally:

```bash
# Install gunicorn
pip install gunicorn

# Run with gunicorn
gunicorn --bind 0.0.0.0:8000 web.app:app
```

Then visit http://localhost:8000

## Features Available After Deployment

- **Main App**: `/` - Camera-based waste detection
- **Dashboard**: `/dashboard` - View all recorded sessions with metrics
- **API**: `/process` - POST endpoint for processing images

## Notes

- Render free tier may have cold starts (30-60 seconds delay after inactivity)
- For better performance, consider upgrading to paid tier
- The app uses OpenCV headless version suitable for servers
- Sessions are automatically saved to Firestore
- Dashboard updates in real-time via Server-Sent Events (SSE)

## Security Best Practices

1. Never commit `.env` file or Firebase JSON to git
2. Use environment variables for all sensitive data
3. Keep your Firebase service account key secure
4. Regularly rotate credentials
5. Set appropriate Firebase security rules

## Monitoring

Monitor your application:
- Render Dashboard → Logs
- Firebase Console → Firestore usage
- Set up alerts for errors or high usage
