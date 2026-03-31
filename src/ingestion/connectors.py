from pycelonis import get_celonis

# Connectors manage the connection to Celonis and provide methods to access different resources (spaces, packages, views, knowledge models, data pools).
class CelonisConnector:
    # Initialization with URL, API token, and key type (USER_KEY or APP_KEY).
    def __init__(self, base_url: str, api_token: str, key_type: str = "USER_KEY"):
        self.base_url = base_url
        self.api_token = api_token
        self.key_type = key_type
        self._celonis = None
        self._studio = None
    
    # Connect to Celonis and initialize the Studio API.
    def connect(self):
        """
        Establish connection to Celonis.
        """
        try:
            self._celonis = get_celonis(
                base_url=self.base_url,
                api_token=self.api_token,
                key_type=self.key_type,
                permissions=False
            )
            self._studio = self._celonis.studio
            return self._celonis
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Celonis: {e}")
    
    # Get space by ID.
    def get_space(self, space_id: str):
        """Get space by ID."""
        if not self._studio:
            self.connect()
        return self._studio.get_space(space_id)
    
    # Get package from space.
    def get_package(self, space_id: str, package_id: str):
        """Get package from space."""
        space = self.get_space(space_id)
        return space.get_package(package_id)
    
    # Get view from package.
    def get_view(self, space_id: str, package_id: str, view_id: str):
        """Get view from package."""
        package = self.get_package(space_id, package_id)
        return package.get_view(view_id)
    
    # Get knowledge model from package.
    def get_knowledge_model(self, space_id: str, package_id: str, km_id: str):
        """Get knowledge model from package."""
        package = self.get_package(space_id, package_id)
        return package.get_knowledge_model(km_id)
    
    # Get data pool by ID or name.
    def get_data_pool(self, pool_identifier: str):
        """Get data pool by ID or name."""
        if not pool_identifier:
            raise ValueError("Data pool identifier must be provided")

        if not self._celonis:
            self.connect()

        # Try lookup by id first (more explicit)
        try:
            pool = self._celonis.data_integration.get_data_pool(pool_identifier)
            if pool:
                return pool
        except Exception:
            # ignore and fallback to name lookup
            pass

        pools = self._celonis.data_integration.get_data_pools()
        pool = pools.find(pool_identifier)
        if not pool:
            raise ValueError(f"Data Pool '{pool_identifier}' not found by id or name")
        return pool
    
    # Get data model by ID or name.
    # In the pycelonis version used here, data models are accessed via the Data Pool object,
    # not via `celonis.datamodels`.
    def get_data_model(self, data_model_id: str, pool_identifier: str = None):
        if not self._celonis:
            self.connect()
        try:
            # If we have a pool id/name, use it (fast path).
            if pool_identifier:
                pool = self.get_data_pool(pool_identifier)
                # Prefer direct getter if available.
                if hasattr(pool, "get_data_model"):
                    datamodel = pool.get_data_model(data_model_id)
                    if datamodel:
                        return datamodel
                # Fallback: search within pool.
                if hasattr(pool, "get_data_models"):
                    models = pool.get_data_models()
                    if models:
                        # Some SDK containers support .find
                        if hasattr(models, "find"):
                            datamodel = models.find(data_model_id)
                            if datamodel:
                                return datamodel
                        # Otherwise iterate
                        for m in models:
                            if getattr(m, "id", None) == data_model_id or getattr(m, "object_id", None) == data_model_id:
                                return m

            # Generic fallback: iterate pools and try to locate the model.
            pools = self._celonis.data_integration.get_data_pools()
            if pools is not None:
                if hasattr(pools, "find"):
                    # If Celonis SDK provides find, still we need pool iteration to search datamodels.
                    # So we fall back to iteration for safety.
                    pass
                for pool in pools:
                    try:
                        if hasattr(pool, "get_data_model"):
                            datamodel = pool.get_data_model(data_model_id)
                            if datamodel:
                                return datamodel
                    except Exception:
                        continue

            raise ValueError(f"Data Model '{data_model_id}' not found in any Data Pool")
        except Exception as e:
            raise ValueError(f"Failed to get data model '{data_model_id}': {e}")
    
    # Test connection by trying to access spaces.
    def test_connection(self) -> bool:
        """Test if connection is working."""
        try:
            celonis = self.connect()
            # Try to get spaces to verify access
            spaces = self._studio.get_spaces()
            return True
        except Exception as e:
            print(f"Connection test failed: {e}")
            return False