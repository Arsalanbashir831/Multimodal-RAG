from django.urls import path
from .views import RegisterView, LoginView, PasswordResetView, VerifyOtpView, FileUploadView, FileListView, FileDeleteView, ChatListCreateView, MessageListCreateView

from .views import PasswordResetConfirmView

urlpatterns = [
    path('register/', RegisterView.as_view(), name='register'),
    path('login/', LoginView.as_view(), name='login'),
    path('password-reset/', PasswordResetView.as_view(), name='password_reset'),
    path('password-reset/confirm/', PasswordResetConfirmView.as_view(), name='password_reset_confirm'),
    path('verify-otp/', VerifyOtpView.as_view(), name='verify_otp'),
    path('files/', FileListView.as_view(), name='file_list'),
    path('files/upload/', FileUploadView.as_view(), name='file_upload'),
    path('files/delete/<int:pk>/', FileDeleteView.as_view(), name='file_delete'),
    path('chats/', ChatListCreateView.as_view(), name='chat_list_create'),
    path('chats/<int:chat_id>/messages/', MessageListCreateView.as_view(), name='message_list_create'),
]