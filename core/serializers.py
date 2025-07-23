from rest_framework import serializers
from .models import File, Chat, Message

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

class  MessageSerializer(serializers.ModelSerializer):
    sender_id = serializers.CharField(read_only=True)
    chat_id = serializers.CharField(read_only=True)

    class Meta:
        model = Message
        fields = ['id', 'chat_id', 'sender_id', 'content', 'timestamp']
        read_only_fields = ['id', 'sender_id', 'timestamp', 'chat_id']