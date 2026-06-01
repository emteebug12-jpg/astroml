"""API routers package."""
from api.routers.fraud import router as fraud_router
from api.routers.accounts import router as accounts_router
from api.routers.monitoring import router as monitoring_router
from api.routers.loyalty import router as loyalty_router

__all__ = ["fraud_router", "accounts_router", "monitoring_router", "loyalty_router"]
