from django.apps import AppConfig


class AppboardConfig(AppConfig):
    name = 'appboard'
    verbose_name = 'appboard'

    def ready(self):
        import appboard.signals
