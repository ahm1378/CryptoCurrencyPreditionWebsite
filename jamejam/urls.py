"""jamejam URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
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
from django.conf.urls.static import static
from trading.views import render_market, render_prediction, draw_history_interview

urlpatterns = [
    path('admin/', admin.site.urls),
    path('market/tradingview/<str:name>/predict/', render_prediction, name="predict"),
    path('market/', render_market, name='market'),
    path('market/tradingview/<str:name>/', draw_history_interview, name='tradingview' )


]
