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
    def get_data_model(self, data_model_id: str):
        data_model_id = data_model_id.strip()
        if not self._celonis:
            self.connect()
        
        # 1. Try to search all pools for the model
        try:
            pools = self._celonis.data_integration.get_data_pools()
            for pool in pools:
                try:
                    dms = pool.get_data_models()
                    # Explicit ID check
                    for dm in dms:
                        if dm.id == data_model_id:
                            return dm
                    # Fallback to find (for names)
                    dm = dms.find(data_model_id)
                    if dm:
                        return dm
                except:
                    continue
        except Exception as e:
            self.logger.debug(f"Search in all pools failed: {e}")
            
        raise ValueError(f"Failed to find Data Model '{data_model_id}' in any accessible pool.")
    
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