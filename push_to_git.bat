@echo off
echo Initializing Git...
git init
git add .
git commit -m "Initial commit of Jewelry Damage Detector app and models"
git branch -M main
echo.
echo Adding remote origin...
git remote add origin https://github.com/leslyvj/Ornament_model.git
echo.
echo Pushing to GitHub...
git push -u origin main
echo.
echo Done! If push failed, check your credentials.
pause
