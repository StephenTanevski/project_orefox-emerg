from django.apps import AppConfig


class BillingConfig(AppConfig):
    name = 'billing'
    verbose_name= 'billing'
    
    def ready(self):
        import billing.signals
