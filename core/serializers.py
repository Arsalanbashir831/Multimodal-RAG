from rest_framework import serializers
from .models import File, Chat, Message

from .models import User

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'email', 'first_name', 'last_name', 'phone_number', 'gender', 'date_of_birth', 'profile_picture']
        read_only_fields = ['id', 'email']

class UserRegistrationSerializer(serializers.Serializer):
    username = serializers.CharField(max_length=150)
    email = serializers.EmailField()
    password = serializers.CharField(write_only=True, required=True, style={'input_type': 'password'})

    def create(self, validated_data):
        # Creation logic should use Supabase Auth client instead
        raise NotImplementedError("Use Supabase Auth client for user creation")

class FileSerializer(serializers.ModelSerializer):
    class Meta:
        model = File
        fields = ['id', 'filename', 'file', 'uploaded_at', 'storage_key']
        read_only_fields = ['id', 'uploaded_at']

class ChatSerializer(serializers.ModelSerializer):
    class Meta:
        model = Chat
        fields = ['id', 'title', 'created_at']
        read_only_fields = ['id', 'created_at']

# serializers.py
from rest_framework import serializers
from .models import Message

class MessageSerializer(serializers.ModelSerializer):
    chat_id     = serializers.IntegerField(read_only=True)
    sender_id   = serializers.CharField(read_only=True)   # or IntegerField if int
    sender_type = serializers.CharField(read_only=True)

    class Meta:
        model  = Message
        fields = ["id", "chat_id", "sender_id", "sender_type", "content", "timestamp"]
        # ONLY mark the fields you set server-side as read-only
        read_only_fields = ["id", "chat_id", "sender_id", "sender_type", "timestamp"]
        # `content` stays writable
