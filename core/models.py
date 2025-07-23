import uuid
from django.db import models
from django.conf import settings
from django.contrib.auth.models import (
    AbstractBaseUser, PermissionsMixin, BaseUserManager
)

class SupabaseUserManager(BaseUserManager):
    def create_user(self, id, email, username=None, **extra_fields):
        """
        We already get the id from Supabase, so we require it.
        Password is not used (Supabase handles auth), but AbstractBaseUser needs something,
        so set unusable password.
        """
        if not id:
            raise ValueError("Supabase user must have an id")
        if not email:
            raise ValueError("Supabase user must have an email")

        email = self.normalize_email(email)
        user = self.model(id=id, email=email, username=username or email.split("@")[0], **extra_fields)
        user.set_unusable_password()
        user.save(using=self._db)
        return user

    def create_superuser(self, email, password=None, **extra_fields):
        """
        Let you create a Django admin superuser manually (no Supabase id). We'll fabricate a UUID.
        """
        extra_fields.setdefault("is_staff", True)
        extra_fields.setdefault("is_superuser", True)

        supa_id = uuid.uuid4().hex
        user = self.model(id=supa_id, email=self.normalize_email(email), username=email.split("@")[0], **extra_fields)
        user.set_password(password or uuid.uuid4().hex)
        user.save(using=self._db)
        return user


class User(AbstractBaseUser, PermissionsMixin):
    # Supabase UUID as primary key (string is fine)
    id = models.CharField(primary_key=True, max_length=255)
    email = models.EmailField(unique=True)
    username = models.CharField(max_length=150, blank=True, null=True)

    # Optional flags for Django admin
    is_active = models.BooleanField(default=True)
    is_staff  = models.BooleanField(default=False)

    created_at = models.DateTimeField(auto_now_add=True)
    last_login = models.DateTimeField(blank=True, null=True)

    objects = SupabaseUserManager()

    USERNAME_FIELD = "email"   # for createsuperuser
    REQUIRED_FIELDS = []       # email is required by default

    def __str__(self):
        return f"{self.email} ({self.id})"


class Chat(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="chats")
    created_at = models.DateTimeField(auto_now_add=True)
    title = models.CharField(max_length=255)

    def __str__(self):
        return f"Chat {self.id} (User {self.user_id})"

class Message(models.Model):
    chat = models.ForeignKey(Chat, on_delete=models.CASCADE, related_name="messages")
    sender = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="messages")
    content = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)

class File(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="files")
    file = models.FileField(upload_to='uploads/')
    filename = models.CharField(max_length=255)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    chroma_collection = models.CharField(max_length=255, blank=True, null=True)  # For ChromaDB collection name/ID
    storage_key = models.CharField(max_length=500, blank=True, null=True)  # Supabase Storage key for file

    def __str__(self):
        return self.filename
