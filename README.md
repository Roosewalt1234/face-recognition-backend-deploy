# Face Recognition Backend - Railway Deployment

This is a standalone FastAPI backend for face recognition attendance system.

## 🚀 Quick Railway Deployment

1. **Create GitHub Repository:**
   - Upload all files from `face-recognition-backend-deploy/` folder
   - Push to a new GitHub repository

2. **Deploy to Railway:**
   - Go to [Railway.app](https://railway.app)
   - Click "New Project" → "Deploy from GitHub repo"
   - Select your repository
   - Railway auto-detects Python and deploys

3. **Set Environment Variables in Railway:**
   ```
   SUPABASE_URL=https://oxuimanrsmredzfdktxg.supabase.co
   SUPABASE_SERVICE_ROLE_KEY=your-actual-service-role-key-here
   SUPABASE_BUCKET=service-images
   MODEL_BUCKET=service-models
   CONFIDENCE_PASS=80
   ```

4. **Get Your API URL:**
   - Railway will give you a URL like: `https://your-app.railway.app`
   - Your API endpoints will be:
     - `https://your-app.railway.app/health`
     - `https://your-app.railway.app/recognize`
     - `https://your-app.railway.app/train`

## 🔧 Update Frontend

Update your frontend `.env` file:
```bash
VITE_API_FACE_RECOG=https://your-app.railway.app
```

## 📋 Files Included

- `main.py` - FastAPI application
- `requirements.txt` - Python dependencies
- `app/` - Core application modules
- `.env` - Environment configuration (update with your values)
- `README.md` - This deployment guide

## 🧪 Test Deployment

After deployment, test with:
```bash
curl https://your-app.railway.app/health
```

Should return: `{"status": "ok", "model_ready": false}`

## 🎯 Next Steps

1. ✅ Deploy this backend
2. ✅ Update frontend with new API URL
3. ✅ Enroll employees using Face Recognition Setup
4. ✅ Train model and test recognition

That's it! Your face recognition backend is ready for deployment! 🚀