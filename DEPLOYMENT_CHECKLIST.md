# Render Deployment Quick Start Checklist

## Pre-Deployment Checklist

- [ ] Firebase Admin SDK JSON file is accessible
- [ ] Git repository created (GitHub, GitLab, or Bitbucket)
- [ ] Render account created (https://render.com)
- [ ] Firestore database is enabled in Firebase Console

## Step-by-Step Deployment

### 1. Convert Firebase Credentials to Base64
```powershell
# Run the helper script
.\convert-firebase-to-base64.ps1
```
Or manually:
```powershell
$fileContent = Get-Content "C:\Users\AIT 33\Documents\Secrets\firebase-admin-sdk.json" -Raw
$bytes = [System.Text.Encoding]::UTF8.GetBytes($fileContent)
$base64 = [Convert]::ToBase64String($bytes)
$base64 | Set-Clipboard
```

- [ ] Base64 string copied to clipboard

### 2. Push Code to Git Repository
```powershell
cd "c:\Users\AIT 33\Desktop\waste2"
git init
git add .
git commit -m "Initial commit for Render deployment"
git branch -M main
git remote add origin <YOUR_REPOSITORY_URL>
git push -u origin main
```

- [ ] Code pushed to remote repository

### 3. Create Web Service on Render

**Option A: Using Blueprint (Easiest)**
1. Go to https://dashboard.render.com
2. Click "New +" → "Blueprint"
3. Connect repository
4. Click "Apply"

**Option B: Manual Setup**
1. Go to https://dashboard.render.com
2. Click "New +" → "Web Service"
3. Connect repository
4. Configure:
   - Name: `waste-volume-estimator`
   - Environment: `Python 3`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn --bind 0.0.0.0:$PORT web.app:app`
   - Plan: Free

- [ ] Web service created on Render

### 4. Configure Environment Variables

In Render Dashboard → Your Service → Environment, add:

| Variable Name | Value | Notes |
|--------------|-------|-------|
| `FLASK_ENV` | `production` | Required |
| `FIREBASE_PROJECT_ID` | `your-project-id` | Find in Firebase Console |
| `FIREBASE_SERVICE_ACCOUNT_BASE64` | `<paste base64 string>` | From Step 1 |

- [ ] `FLASK_ENV` added
- [ ] `FIREBASE_PROJECT_ID` added
- [ ] `FIREBASE_SERVICE_ACCOUNT_BASE64` added

### 5. Deploy Application

1. Click "Manual Deploy" → "Deploy latest commit"
2. Wait for build to complete (5-10 minutes)
3. Monitor logs for any errors

- [ ] Deployment started
- [ ] Build completed successfully
- [ ] Application is live

### 6. Verify Deployment

1. Visit your Render URL: `https://your-service-name.onrender.com`
2. Test main page (`/`)
3. Test dashboard (`/dashboard`)
4. Try capturing and processing an image
5. Verify data appears in Firestore

- [ ] Main page loads
- [ ] Dashboard loads
- [ ] Image processing works
- [ ] Data saved to Firestore

### 7. Firebase Firestore Setup

If not already set up:
1. Go to Firebase Console (https://console.firebase.google.com)
2. Select your project
3. Navigate to "Firestore Database"
4. Click "Create database"
5. Choose "Start in production mode" or "Start in test mode"
6. Select a location (choose closest to your users)

- [ ] Firestore database created
- [ ] Database rules configured (if needed)

## Important URLs

- **Render Dashboard**: https://dashboard.render.com
- **Firebase Console**: https://console.firebase.google.com
- **Your App URL**: `https://your-service-name.onrender.com`
- **Dashboard URL**: `https://your-service-name.onrender.com/dashboard`

## Common Issues & Solutions

### Issue: Build fails with OpenCV error
**Solution**: Already fixed! We use `opencv-python-headless` in requirements.txt

### Issue: Firebase authentication error
**Solution**: Verify base64 encoding and FIREBASE_PROJECT_ID are correct

### Issue: App crashes on startup
**Solution**: Check logs in Render dashboard, ensure all env variables are set

### Issue: Cold start takes long
**Solution**: Normal for free tier (30-60 seconds). Consider upgrading plan.

### Issue: "Firestore not initialized" error
**Solution**: Create Firestore database in Firebase Console

## Next Steps After Deployment

1. Set up custom domain (optional)
2. Configure Firebase security rules
3. Set up monitoring/alerts
4. Consider upgrading to paid plan for better performance
5. Add health check endpoint (optional)

## Environment Variables Reference

```bash
# Production (Render)
FLASK_ENV=production
FIREBASE_PROJECT_ID=your-firebase-project-id
FIREBASE_SERVICE_ACCOUNT_BASE64=<base64-encoded-json>

# Local Development
FLASK_ENV=development
PORT=8000
FIREBASE_CREDENTIALS_FILE=C:\Users\AIT 33\Documents\Secrets\firebase-admin-sdk.json
```

## Support & Resources

- Render Documentation: https://render.com/docs
- Firebase Admin SDK: https://firebase.google.com/docs/admin/setup
- Flask Deployment: https://flask.palletsprojects.com/en/latest/deploying/
