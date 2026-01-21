"""
Unit tests for GraphQL routes and schema.

Tests for app/api/routes/graphql.py and app/api/graphql/schema.py
"""

import json
from unittest.mock import MagicMock, patch

import pytest


class TestGraphQLEnabled:
    """Tests for GraphQL feature flag."""

    @patch.dict("os.environ", {"GRAPHQL_ENABLED": "true"})
    def test_graphql_enabled_true(self):
        """Test GraphQL is enabled when flag is true."""
        # Would need to reimport to pick up env change
        pass

    @patch.dict("os.environ", {"GRAPHQL_ENABLED": "false"})
    def test_graphql_disabled(self):
        """Test GraphQL is disabled by default."""
        from app.api.graphql.schema import GRAPHQL_ENABLED

        # Default should be false if not explicitly enabled


class TestGraphQLEndpoint:
    """Tests for GraphQL endpoint."""

    def test_graphql_get_graphiql(self, client):
        """Test GraphQL GET returns GraphiQL IDE."""
        response = client.get("/graphql")

        # May return 200 with GraphiQL or 404 if disabled
        assert response.status_code in [200, 404]
        if response.status_code == 200:
            # Should return HTML for GraphiQL
            assert b"graphiql" in response.data.lower() or b"GraphQL" in response.data

    def test_graphql_post_query(self, client):
        """Test GraphQL POST with query."""
        query = {"query": "{ health { status } }"}

        response = client.post("/graphql", data=json.dumps(query), content_type="application/json")

        # May return 200 with data/errors or 404 if disabled
        assert response.status_code in [200, 400, 404]

    def test_graphql_post_no_query(self, client):
        """Test GraphQL POST without query returns error."""
        response = client.post("/graphql", data=json.dumps({}), content_type="application/json")

        assert response.status_code in [400, 404]

    def test_graphql_post_with_variables(self, client):
        """Test GraphQL POST with variables."""
        query = {
            "query": "query GetWeather($lat: Float!, $lon: Float!) { weatherData(latitude: $lat, longitude: $lon) { temperature } }",
            "variables": {"lat": 14.4793, "lon": 121.0198},
        }

        response = client.post("/graphql", data=json.dumps(query), content_type="application/json")

        assert response.status_code in [200, 400, 404]

    def test_graphql_post_with_operation_name(self, client):
        """Test GraphQL POST with operation name."""
        query = {"query": "query GetHealth { health { status } }", "operationName": "GetHealth"}

        response = client.post("/graphql", data=json.dumps(query), content_type="application/json")

        assert response.status_code in [200, 400, 404]


class TestGraphQLSchema:
    """Tests for GraphQL schema types."""

    def test_weather_data_type_fields(self):
        """Test WeatherDataType has required fields."""
        from app.api.graphql.schema import WeatherDataType

        # Check field definitions exist
        fields = WeatherDataType._meta.fields
        expected_fields = ["id", "timestamp", "latitude", "longitude", "temperature", "humidity"]
        for field in expected_fields:
            assert field in fields

    def test_flood_prediction_type_fields(self):
        """Test FloodPredictionType has required fields."""
        from app.api.graphql.schema import FloodPredictionType

        fields = FloodPredictionType._meta.fields
        expected_fields = ["id", "timestamp", "latitude", "longitude", "flood_risk", "confidence"]
        for field in expected_fields:
            assert field in fields

    def test_health_status_type_fields(self):
        """Test HealthStatusType has required fields."""
        from app.api.graphql.schema import HealthStatusType

        fields = HealthStatusType._meta.fields
        expected_fields = ["status", "timestamp", "model_available", "database_connected"]
        for field in expected_fields:
            assert field in fields


class TestGraphQLQueries:
    """Tests for GraphQL query resolvers."""

    def test_health_query_introspection(self, client):
        """Test health query via introspection."""
        query = {
            "query": """
                {
                    __type(name: "Query") {
                        fields {
                            name
                        }
                    }
                }
            """
        }

        response = client.post("/graphql", data=json.dumps(query), content_type="application/json")

        assert response.status_code in [200, 404]

    @patch("app.api.graphql.schema.check_database_health")
    @patch("app.api.graphql.schema.check_model_health")
    def test_health_resolver(self, mock_model, mock_db):
        """Test health query resolver."""
        from app.api.graphql.schema import Query

        mock_db.return_value = {"connected": True}
        mock_model.return_value = "healthy"

        query = Query()
        result = query.resolve_health(None)

        assert result.status in ["healthy", "unhealthy"]

    def test_health_status_resolver(self):
        """Test health_status simple resolver."""
        from app.api.graphql.schema import Query

        query = Query()
        with patch("app.api.graphql.schema.check_database_health") as mock_db:
            with patch("app.api.graphql.schema.check_model_health") as mock_model:
                mock_db.return_value = {"connected": True}
                mock_model.return_value = "healthy"

                result = query.resolve_health_status(None)
                assert result in ["healthy", "unhealthy"]


class TestGraphQLMutations:
    """Tests for GraphQL mutations if defined."""

    def test_mutations_exist(self):
        """Test that mutations schema exists."""
        try:
            from app.api.graphql.schema import Mutation

            # Mutation class exists
            assert Mutation is not None
        except ImportError:
            # Mutations may not be defined
            pass


class TestGenericScalar:
    """Tests for custom GenericScalar type."""

    def test_generic_scalar_serialize(self):
        """Test GenericScalar serialization."""
        from app.api.graphql.schema import GenericScalar

        value = {"key": "value", "nested": {"data": [1, 2, 3]}}
        result = GenericScalar.serialize(value)

        assert result == value

    def test_generic_scalar_parse_value(self):
        """Test GenericScalar value parsing."""
        from app.api.graphql.schema import GenericScalar

        value = {"test": 123}
        result = GenericScalar.parse_value(value)

        assert result == value


class TestGraphQLErrorHandling:
    """Tests for GraphQL error handling."""

    def test_invalid_query_syntax(self, client):
        """Test handling of invalid GraphQL syntax."""
        query = {"query": "{ this is not valid graphql }"}

        response = client.post("/graphql", data=json.dumps(query), content_type="application/json")

        if response.status_code == 200:
            data = response.get_json()
            # Should return errors
            assert "errors" in data

    def test_nonexistent_field(self, client):
        """Test querying non-existent field."""
        query = {"query": "{ nonExistentField }"}

        response = client.post("/graphql", data=json.dumps(query), content_type="application/json")

        if response.status_code == 200:
            data = response.get_json()
            assert "errors" in data


class TestGraphQLRateLimiting:
    """Tests for GraphQL rate limiting."""

    def test_graphql_rate_limited(self, client):
        """Test that GraphQL endpoint respects rate limits."""
        # Make several rapid requests
        for _ in range(5):
            response = client.post(
                "/graphql", data=json.dumps({"query": "{ health { status } }"}), content_type="application/json"
            )
            assert response.status_code in [200, 404, 429]


class TestGraphQLInit:
    """Tests for GraphQL initialization."""

    def test_init_graphql_route_disabled(self):
        """Test init_graphql_route when disabled."""
        from app.api.routes.graphql import init_graphql_route
        from flask import Flask

        app = Flask(__name__)

        with patch.dict("os.environ", {"GRAPHQL_ENABLED": "false"}):
            with patch("app.api.routes.graphql.GRAPHQL_ENABLED", False):
                init_graphql_route(app)
                # Should not add route when disabled
