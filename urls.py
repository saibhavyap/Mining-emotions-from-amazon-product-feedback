"""cyber URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static

from cyber import views as admins
from user import views as usr

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', admins.index, name='index'),
    path('Home/', admins.Home, name='Home'),
    path('adminlogin/', admins.adminlogin, name='adminlogin'),
    path('adminloginaction/', admins.adminloginaction, name='adminloginaction'),
    path('showusers/', admins.showusers, name='showusers'),
    path('AdminActiveUsers/', admins.AdminActiveUsers, name='AdminActiveUsers'),
    path('ML/',admins.ML, name='ML'),
    path('logout/', admins.logout, name='logout'),

    path('Userlogin/', usr.Userlogin, name='Userlogin'),
    path('userregister/', usr.userregister, name='userregister'),
    path('userregisterAction/', usr.userregisterAction, name='userregisterAction'),
    path('userloginaction/', usr.userloginaction, name='userloginaction'),
    path('predict/', usr.predict, name='predict'),
    path('predicts/', usr.predicts, name='predicts'),
    path('usrlogout/', usr.usrlogout, name='usrlogout'),
]
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)