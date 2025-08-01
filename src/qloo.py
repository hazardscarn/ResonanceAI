import requests
import json
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from urllib.parse import urlencode

@dataclass
class QlooSignals:
    """Structure for Qloo API signals"""
    demographics: Optional[Dict[str, str]] = None  # {"age": "25_to_29", "gender": "male"}
    location: Optional[Dict[str, str]] = None      # {"query": "New York"}
    entity_ids: Optional[List[str]] = None         # List of entity IDs for context
    entity_queries: Optional[List[Union[str, Dict[str, str]]]] = None  # Text queries for entities
    tag_ids: Optional[List[str]] = None            # List of tag IDs for context
    audience_ids: Optional[List[str]] = None       # NEW: List of audience IDs for signals
    audience_weight: Optional[float] = None        # NEW: Weight for audience influence (0-1)

@dataclass
class QlooAudience:
    """Structure for Qloo audience data"""
    id: str
    name: str
    entity_id: Optional[str] = None
    parent_type: Optional[str] = None

@dataclass
class QlooEntity:
    """Structure for Qloo entity search results"""
    id: str
    name: str
    entity_type: str
    type_urn: str
    affinity_score: Optional[float] = None
    popularity: Optional[float] = None
    additional_info: Optional[Dict[str, Any]] = None

class QlooAPIClient:
    """
    Clean API client for Qloo Hackathon API
    Handles audience discovery, insights extraction, demographics analysis, and heatmap analysis
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key.strip()
        self.base_url = "https://hackathon.api.qloo.com"
        self.headers = {
            "X-Api-Key": self.api_key,
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        # Supported audience parent types
        self.audience_types = [
            "urn:audience:hobbies_and_interests",
            "urn:audience:spending_habits",
            "urn:audience:communities",
            "urn:audience:global_issues",
            "urn:audience:investing_interests",
            "urn:audience:leisure",
            "urn:audience:life_stage",
            "urn:audience:lifestyle_preferences_beliefs",
            "urn:audience:political_preferences",
            "urn:audience:professional_area"
        ]
        
        # Supported entity types for insights (excluding tags)
        self.entity_types = {
            "brand": "urn:entity:brand",
            "artist": "urn:entity:artist",
            "movie": "urn:entity:movie", 
            "place": "urn:entity:place",
            "book": "urn:entity:book",
            "people": "urn:entity:person",
            "podcast": "urn:entity:podcast",
            "videogame": "urn:entity:videogame",
            "tv_show": "urn:entity:tv_show"
        }
    
    def _build_url(self, endpoint: str, params: Dict[str, str] = None, encode: bool = True) -> str:
        """Build full URL with parameters for Postman testing"""
        if params is None:
            return f"{self.base_url}{endpoint}"
            
        if encode:
            query_string = urlencode(params)
        else:
            # Build unencoded URL for easier Postman testing
            query_parts = [f"{key}={value}" for key, value in params.items()]
            query_string = "&".join(query_parts)
        
        return f"{self.base_url}{endpoint}?{query_string}"

    def _build_readable_url(self, endpoint: str, params: Dict[str, str]) -> str:
        """Build human-readable URL for Postman (unencoded)"""
        return self._build_url(endpoint, params, encode=False)
    
    def _should_use_post(self, signals: QlooSignals) -> bool:
        """Determine if POST request is needed (required for entity queries on insights endpoint only)"""
        return signals and signals.entity_queries is not None
    
    def _build_post_body(self, params: Dict[str, Any], signals: Optional[QlooSignals] = None) -> Dict[str, Any]:
        """Build POST request body from parameters and signals"""
        body = params.copy()
        
        if signals and signals.entity_queries:
            # Convert entity queries to proper format
            entity_queries = []
            for query in signals.entity_queries:
                if isinstance(query, str):
                    # Simple string query - resolve to both place and brand by default
                    entity_queries.append({
                        "name": query,
                        "resolve_to": "both"
                    })
                elif isinstance(query, dict):
                    # Already formatted query object
                    entity_queries.append(query)
            
            body["signal.interests.entities.query"] = entity_queries
            
            # Remove the regular entity_ids if present since query overwrites it
            if "signal.interests.entities" in body:
                del body["signal.interests.entities"]
        
        return body
    
    def _add_signal_params(self, params: Dict[str, str], signals: Optional[QlooSignals]) -> None:
        """Add signal parameters to params dict"""
        if not signals:
            return
            
        # Add demographic signals
        if signals.demographics:
            if "age" in signals.demographics:
                params["signal.demographics.age"] = signals.demographics["age"]
            if "gender" in signals.demographics:
                params["signal.demographics.gender"] = signals.demographics["gender"]
        
        # Add location signals
        if signals.location and "query" in signals.location:
            params["signal.location.query"] = signals.location["query"]
        
        # Add entity IDs
        if signals.entity_ids:
            params["signal.interests.entities"] = ",".join(signals.entity_ids)
            
        # Add tag IDs
        if signals.tag_ids:
            params["signal.interests.tags"] = ",".join(signals.tag_ids)
        
        # NEW: Add audience signals
        if signals.audience_ids:
            params["signal.demographics.audiences"] = ",".join(signals.audience_ids)
        
        # NEW: Add audience weight
        if signals.audience_weight is not None:
            params["signal.demographics.audiences.weight"] = str(signals.audience_weight)

    def search_entities(
        self,
        query: str,
        entity_types: Optional[List[str]] = None,
        signals: Optional[QlooSignals] = None,
        limit: int = 5,
        sort_by: str = "match",
        min_popularity: Optional[float] = None,
        min_rating: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Search for entities by name/query with optional filters and signals
        
        Args:
            query: The text to search for (e.g., "Chelsea FC", "Nike", "Taylor Swift")
            entity_types: List of entity types to search in (e.g., ["brand", "artist", "place"])
            signals: Additional signals to apply (demographics, location, etc.)
            limit: Maximum number of results to return (1-100)
            sort_by: Sort criteria ("match", "distance", "popularity")
            min_popularity: Minimum popularity score (0-1)
            min_rating: Minimum rating (0-5)
            
        Returns:
            Dictionary with search results and metadata
        """
        
        if not query.strip():
            return {
                "success": False,
                "error": "Query parameter is required",
                "entities": [],
                "total_found": 0
            }
        
        # Build query parameters
        params = {
            "query": query.strip(),
            "take": str(min(limit, 100)),  # API max is 100
            "sort_by": sort_by
        }
        
        # Add entity type filters
        if entity_types:
            type_urns = []
            for entity_type in entity_types:
                if entity_type in self.entity_types:
                    type_urns.append(self.entity_types[entity_type])
                elif entity_type.startswith("urn:entity:"):
                    type_urns.append(entity_type)  # Allow direct URNs
                else:
                    print(f"⚠️ Unknown entity type: {entity_type}")
            
            if type_urns:
                params["filter.types"] = ",".join(type_urns)
        
        # Add optional filters
        if min_popularity is not None:
            params["filter.popularity"] = str(min_popularity)
        if min_rating is not None:
            params["filter.rating"] = str(min_rating)
        
        # Add signal filters
        if signals:
            # Demographics signals
            if signals.demographics:
                if "age" in signals.demographics:
                    params["signal.demographics.age"] = signals.demographics["age"]
                if "gender" in signals.demographics:
                    params["signal.demographics.gender"] = signals.demographics["gender"]
            
            # Location signals  
            if signals.location and "query" in signals.location:
                params["signal.location.query"] = signals.location["query"]
        
        endpoint = "/search"
        postman_url = self._build_readable_url(endpoint, params)
        
        try:
            response = requests.get(
                f"{self.base_url}{endpoint}",
                headers=self.headers,
                params=params,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                entities = []
                
                # Parse results
                if "results" in data:
                    for result in data["results"]:
                        # Extract entity information
                        entity_id = result.get("entity_id", "")
                        entity_name = result.get("name", "Unknown")
                        
                        # Fix: API returns "types" (array) not "type" (string)
                        entity_types_array = result.get("types", [])
                        entity_type_urn = entity_types_array[0] if entity_types_array else ""
                        
                        # Convert URN to readable type - handle various URN formats
                        readable_type = "unknown"
                        if entity_type_urn:
                            # Direct mapping first
                            for key, urn in self.entity_types.items():
                                if urn == entity_type_urn:
                                    readable_type = key
                                    break
                            
                            # If no direct match, try extracting from URN pattern
                            if readable_type == "unknown":
                                if "urn:entity:" in entity_type_urn:
                                    urn_suffix = entity_type_urn.replace("urn:entity:", "")
                                    # Map common variations
                                    type_mappings = {
                                        "brand": "brand",
                                        "person": "people", 
                                        "artist": "artist",
                                        "movie": "movie",
                                        "tv_show": "tv_show",
                                        "place": "place",
                                        "book": "book",
                                        "podcast": "podcast",
                                        "videogame": "videogame"
                                    }
                                    readable_type = type_mappings.get(urn_suffix, urn_suffix)
                        
                        entity = QlooEntity(
                            id=entity_id,
                            name=entity_name,
                            entity_type=readable_type,
                            type_urn=entity_type_urn,
                            affinity_score=result.get("affinity_score"),
                            popularity=result.get("popularity"),
                            additional_info={
                                k: v for k, v in result.items() 
                                if k not in ["entity_id", "name", "type", "affinity_score", "popularity"]
                            }
                        )
                        entities.append(entity)
                
                return {
                    "success": True,
                    "query": query,
                    "entity_types_searched": entity_types or "all",
                    "entities": entities,
                    "total_found": len(entities),
                    "signals_used": {
                        "demographics": signals.demographics if signals and signals.demographics else None,
                        "location": signals.location if signals and signals.location else None
                    },
                    "params_used": params,
                    "raw_response": data,
                    "postman_url": postman_url,
                    "headers_needed": {"X-Api-Key": "YOUR_API_KEY"}
                }
                
            else:
                return {
                    "success": False,
                    "query": query,
                    "entity_types_searched": entity_types or "all",
                    "entities": [],
                    "total_found": 0,
                    "error": f"API Error {response.status_code}: {response.text}",
                    "status_code": response.status_code,
                    "params_used": params,
                    "postman_url": postman_url,
                    "headers_needed": {"X-Api-Key": "YOUR_API_KEY"}
                }
                
        except Exception as e:
            return {
                "success": False,
                "query": query,
                "entity_types_searched": entity_types or "all",
                "entities": [],
                "total_found": 0,
                "error": f"Search request failed: {str(e)}",
                "params_used": params,
                "postman_url": postman_url,
                "headers_needed": {"X-Api-Key": "YOUR_API_KEY"}
            }
    
    def get_entity_info(
        self,
        queries: List[str],
        entity_types: Optional[List[str]] = None,
        signals: Optional[QlooSignals] = None,
        limit_per_query: int = 5
    ) -> Dict[str, Any]:
        """
        Search for multiple entities and return the top results for each query
        
        Args:
            queries: List of search queries (e.g., ["Chelsea FC", "Nike", "Taylor Swift"])
            entity_types: List of entity types to search in (optional)
            signals: Additional signals to apply (demographics, location, etc.)
            limit_per_query: Maximum results per query
            
        Returns:
            Dictionary with results for each query
        """
        
        if not queries:
            return {
                "success": False,
                "error": "At least one query is required",
                "query_results": {},
                "total_entities_found": 0
            }
        
        results = {}
        total_found = 0
        all_entities = []
        postman_urls = {}
        
        print(f"🔍 Searching for {len(queries)} entities...")
        
        for i, query in enumerate(queries):
            if not query.strip():
                continue
                
            print(f"  📍 Query {i+1}/{len(queries)}: '{query}'")
            
            search_result = self.search_entities(
                query=query,
                entity_types=entity_types,
                signals=signals,
                limit=limit_per_query
            )
            
            results[query] = search_result
            postman_urls[query] = search_result.get("postman_url", "")
            
            if search_result.get("success"):
                entities = search_result.get("entities", [])
                total_found += len(entities)
                all_entities.extend(entities)
                
                # Print results
                if entities:
                    print(f"    ✅ Found {len(entities)} entities:")
                    for entity in entities[:3]:  # Show top 3
                        score_info = f" (affinity: {entity.affinity_score:.3f})" if entity.affinity_score else ""
                        print(f"      • {entity.name} [{entity.entity_type}]{score_info}")
                    if len(entities) > 3:
                        print(f"      ... and {len(entities) - 3} more")
                else:
                    print(f"    ❌ No entities found")
            else:
                print(f"    ❌ Search failed: {search_result.get('error', 'Unknown error')}")
        
        return {
            "success": total_found > 0,
            "queries": queries,
            "entity_types_searched": entity_types or "all",
            "signals_used": {
                "demographics": signals.demographics if signals and signals.demographics else None,
                "location": signals.location if signals and signals.location else None
            },
            "query_results": results,
            "all_entities": all_entities,
            "total_entities_found": total_found,
            "summary": {
                "queries_processed": len([q for q in queries if q.strip()]),
                "successful_queries": len([q for q, r in results.items() if r.get("success")]),
                "total_entities": total_found
            },
            "postman_urls": postman_urls,
            "headers_needed": {"X-Api-Key": "YOUR_API_KEY"}
        }

    def test_connection(self) -> Dict[str, Any]:
        """Test API connection"""
        params = {"filter.type": "urn:entity:brand", "take": "5", "signal.location.query": "North America"}
        endpoint = "/v2/insights"
        test_url = self._build_readable_url(endpoint, params)
        
        try:
            response = requests.get(
                f"{self.base_url}{endpoint}",
                headers=self.headers,
                params=params,
                timeout=10
            )
            
            return {
                "success": response.status_code == 200,
                "status_code": response.status_code,
                "message": "Connection successful" if response.status_code == 200 else f"Error: {response.text}",
                "postman_url": test_url,
                "headers_needed": {"X-Api-Key": "YOUR_API_KEY"}
            }
            
        except Exception as e:
            return {
                "success": False,
                "status_code": None,
                "message": f"Connection failed: {str(e)}",
                "postman_url": test_url,
                "headers_needed": {"X-Api-Key": "YOUR_API_KEY"}
            }
    
    def _resolve_entity_queries_to_ids(self, entity_queries: List[Union[str, Dict[str, str]]]) -> List[str]:
        """
        Resolve text-based entity queries to entity IDs using the insights endpoint
        This is needed because the audiences endpoint doesn't support entity queries
        """
        if not entity_queries:
            return []
        
        # Build a quick insights request to resolve entities
        body = {
            "filter.type": "urn:entity:brand",  # Use brands as a simple entity type
            "take": "1",  # We only need to resolve entities, not get results
            "signal.interests.entities.query": []
        }
        
        # Format entity queries
        for query in entity_queries:
            if isinstance(query, str):
                body["signal.interests.entities.query"].append({
                    "name": query,
                    "resolve_to": "both"
                })
            elif isinstance(query, dict):
                body["signal.interests.entities.query"].append(query)
        
        try:
            response = requests.post(
                f"{self.base_url}/v2/insights",
                headers=self.headers,
                json=body,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                entity_ids = []
                
                # Extract resolved entity IDs
                query_info = data.get("query", {})
                if "entities" in query_info and "signal" in query_info["entities"]:
                    for entity in query_info["entities"]["signal"]:
                        entity_id = entity.get("entity_id")
                        if entity_id:
                            entity_ids.append(entity_id)
                            entity_name = entity.get("name", "Unknown")
                            print(f"✅ Resolved entity query -> {entity_name} ({entity_id})")
                
                return entity_ids
            else:
                print(f"⚠️ Failed to resolve entity queries: {response.status_code} - {response.text}")
                return []
                
        except Exception as e:
            print(f"⚠️ Error resolving entity queries: {str(e)}")
            return []

    def find_audiences(self, signals: QlooSignals, audience_types=None, limit: int = 50) -> Dict[str, Any]:
        """
        Find audiences based on provided signals
        
        Args:
            signals: QlooSignals object with demographics, location, etc.
            audience_types: List of audience types to search (optional)
            limit: Maximum number of audiences to return
            
        Returns:
            Dictionary with audiences and URLs for testing
        """
        all_audiences = []
        postman_urls = {}
        resolved_entities = []
        
        # IMPORTANT: Audiences endpoint doesn't support entity queries!
        # We need to resolve queries to entity IDs first
        if signals and signals.entity_queries:
            print("🔄 Resolving entity queries to entity IDs (audiences endpoint requires entity IDs)...")
            resolved_entity_ids = self._resolve_entity_queries_to_ids(signals.entity_queries)
            
            if resolved_entity_ids:
                # Create a new signals object with resolved entity IDs
                signals = QlooSignals(
                    demographics=signals.demographics,
                    location=signals.location,
                    entity_ids=resolved_entity_ids,
                    entity_queries=None  # Clear queries since we resolved them
                )
                resolved_entities = resolved_entity_ids
                print(f"✅ Resolved {len(resolved_entity_ids)} entities for audience search")
            else:
                print("❌ No entities could be resolved from your queries")
                return {
                    "audiences": [],
                    "postman_urls": {},
                    "headers_needed": {"X-Api-Key": "YOUR_API_KEY"},
                    "total_found": 0,
                    "resolved_entities": [],
                    "warning": "No entities could be resolved from your queries"
                }
        
        if audience_types is None:
            audience_types = self.audience_types

        for parent_type in audience_types:
            # Build query parameters - audiences endpoint only supports GET with entity IDs
            params = {
                "filter.parents.types": parent_type,
                "take": str(limit)
            }
            
            # Add demographic signals
            if signals.demographics:
                if "age" in signals.demographics:
                    params["signal.demographics.age"] = signals.demographics["age"]
                if "gender" in signals.demographics:
                    params["signal.demographics.gender"] = signals.demographics["gender"]
            
            # Add location signals
            if signals.location and "query" in signals.location:
                params["signal.location.query"] = signals.location["query"]
            
            # Add entity IDs (resolved from queries or provided directly)
            if signals.entity_ids:
                params["signal.interests.entities"] = ",".join(signals.entity_ids)
            
            endpoint = "/v2/audiences"
            postman_url = self._build_readable_url(endpoint, params)
            postman_urls[parent_type] = postman_url
            
            try:
                response = requests.get(
                    f"{self.base_url}{endpoint}",
                    headers=self.headers,
                    params=params,
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if "results" in data and "audiences" in data["results"]:
                        audiences = data["results"]["audiences"]
                        
                        for aud in audiences:
                            audience = QlooAudience(
                                id=aud.get("id", ""),
                                name=aud.get("name", "Unknown"),
                                entity_id=aud.get("entity_id"),
                                parent_type=parent_type
                            )
                            all_audiences.append(audience)
                            
                elif response.status_code != 404:  # 404 is expected for some combinations
                    print(f"Warning: Audience search failed for {parent_type}: {response.status_code}")
                    
            except Exception as e:
                print(f"Error searching audiences for {parent_type}: {str(e)}")
        
        # Remove duplicates
        seen_ids = set()
        unique_audiences = []
        for aud in all_audiences:
            if aud.id not in seen_ids:
                seen_ids.add(aud.id)
                unique_audiences.append(aud)
        
        return {
            "audiences": unique_audiences[:limit],
            "postman_urls": postman_urls,
            "headers_needed": {"X-Api-Key": "YOUR_API_KEY"},
            "total_found": len(unique_audiences),
            "resolved_entities": resolved_entities
        }
    
    def get_entity_insights(
        self, 
        audience_ids: List[str], 
        entity_type: str = "brands",
        signals: Optional[QlooSignals] = None,
        limit: int = 20
    ) -> Dict[str, Any]:
        """
        Get entity insights for given audience IDs
        
        Args:
            audience_ids: List of audience IDs to get insights for
            entity_type: Type of entities to get ("brands", "artists", "movies", "places", "books", "people", "podcasts", "videogames")
            signals: Additional signals to apply (demographics, location, entity queries)
            limit: Maximum number of results to return
            
        Returns:
            Dictionary with entity insights data and Postman URL
        """
        
        # Validate entity type
        if entity_type not in self.entity_types:
            raise ValueError(f"Unsupported entity type: {entity_type}. Use: {list(self.entity_types.keys())}")
        
        # Build query parameters
        params = {
            "filter.type": self.entity_types[entity_type],
            "signal.demographics.audiences": ",".join(audience_ids),
            "take": str(limit)
        }
        
        # Add additional signals if provided
        self._add_signal_params(params, signals)
        
        # Check if we need POST request
        use_post = self._should_use_post(signals)
        endpoint = "/v2/insights"
        
        if use_post:
            # Use POST request with JSON body
            body = self._build_post_body(params, signals)
            
            try:
                response = requests.post(
                    f"{self.base_url}{endpoint}",
                    headers=self.headers,
                    json=body,
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return {
                        "success": True,
                        "entity_type": entity_type,
                        "audience_ids": audience_ids,
                        "request_method": "POST",
                        "params_used": body,
                        "results": data.get("results", {}),
                        "query_info": data.get("query", {}),  # Contains resolved entities from queries
                        "raw_response": data,
                        "postman_url": f"{self.base_url}{endpoint} (POST)",
                        "postman_body": body,
                        "headers_needed": {"X-Api-Key": "YOUR_API_KEY"}
                    }
                else:
                    return {
                        "success": False,
                        "entity_type": entity_type,
                        "audience_ids": audience_ids,
                        "request_method": "POST",
                        "params_used": body,
                        "error": f"API Error {response.status_code}: {response.text}",
                        "status_code": response.status_code,
                        "postman_url": f"{self.base_url}{endpoint} (POST)",
                        "postman_body": body,
                        "headers_needed": {"X-Api-Key": "YOUR_API_KEY"}
                    }
            except Exception as e:
                return {
                    "success": False,
                    "entity_type": entity_type,
                    "audience_ids": audience_ids,
                    "request_method": "POST",
                    "error": f"POST request failed: {str(e)}",
                    "postman_url": f"{self.base_url}{endpoint} (POST)",
                    "postman_body": body,
                    "headers_needed": {"X-Api-Key": "YOUR_API_KEY"}
                }
        else:
            # Use GET request
            postman_url = self._build_readable_url(endpoint, params)
            
            try:
                response = requests.get(
                    f"{self.base_url}{endpoint}",
                    headers=self.headers,
                    params=params,
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return {
                        "success": True,
                        "entity_type": entity_type,
                        "audience_ids": audience_ids,
                        "request_method": "GET",
                        "params_used": params,
                        "results": data.get("results", {}),
                        "raw_response": data,
                        "postman_url": postman_url,
                        "headers_needed": {"X-Api-Key": "YOUR_API_KEY"}
                    }
                else:
                    return {
                        "success": False,
                        "entity_type": entity_type,
                        "audience_ids": audience_ids,
                        "request_method": "GET",
                        "params_used": params,
                        "error": f"API Error {response.status_code}: {response.text}",
                        "status_code": response.status_code,
                        "postman_url": postman_url,
                        "headers_needed": {"X-Api-Key": "YOUR_API_KEY"}
                    }
            except Exception as e:
                return {
                    "success": False,
                    "entity_type": entity_type,
                    "audience_ids": audience_ids,
                    "request_method": "GET",
                    "error": f"GET request failed: {str(e)}",
                    "postman_url": postman_url,
                    "headers_needed": {"X-Api-Key": "YOUR_API_KEY"}
                }
    
    def get_tag_insights(
        self,
        audience_ids: List[str],
        signals: Optional[QlooSignals] = None,
        limit: int = 20,
        tag_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get tag insights for given audience IDs
        
        Args:
            audience_ids: List of audience IDs to get insights for
            signals: Additional signals to apply (demographics, location, entity queries)
            limit: Maximum number of results to return
            tag_filter: Optional specific tag filter (e.g., "urn:tag:lifestyle:qloo")
            
        Returns:
            Dictionary with tag insights data and Postman URL
        """
        
        # Build query parameters
        params = {
            "filter.type": "urn:tag",
            "signal.demographics.audiences": ",".join(audience_ids),
            "take": str(limit)
        }
        
        # Add specific tag filter if provided
        if tag_filter:
            params["filter.tag.types"] = tag_filter
        
        # Add additional signals if provided
        self._add_signal_params(params, signals)
        
        # Check if we need POST request
        use_post = self._should_use_post(signals)
        endpoint = "/v2/insights"
        
        if use_post:
            # Use POST request with JSON body
            body = self._build_post_body(params, signals)
            
            try:
                response = requests.post(
                    f"{self.base_url}{endpoint}",
                    headers=self.headers,
                    json=body,
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return {
                        "success": True,
                        "tag_filter": tag_filter or "all_tags",
                        "audience_ids": audience_ids,
                        "request_method": "POST",
                        "params_used": body,
                        "results": data.get("results", {}),
                        "query_info": data.get("query", {}),
                        "raw_response": data,
                        "postman_url": f"{self.base_url}{endpoint} (POST)",
                        "postman_body": body,
                        "headers_needed": {"X-Api-Key": "YOUR_API_KEY"}
                    }
                else:
                    return {
                        "success": False,
                        "tag_filter": tag_filter or "all_tags",
                        "audience_ids": audience_ids,
                        "request_method": "POST",
                        "params_used": body,
                        "error": f"API Error {response.status_code}: {response.text}",
                        "status_code": response.status_code,
                        "postman_url": f"{self.base_url}{endpoint} (POST)",
                        "postman_body": body,
                        "headers_needed": {"X-Api-Key": "YOUR_API_KEY"}
                    }
            except Exception as e:
                return {
                    "success": False,
                    "tag_filter": tag_filter or "all_tags",
                    "audience_ids": audience_ids,
                    "request_method": "POST",
                    "error": f"POST request failed: {str(e)}",
                    "postman_url": f"{self.base_url}{endpoint} (POST)",
                    "postman_body": body,
                    "headers_needed": {"X-Api-Key": "YOUR_API_KEY"}
                }
        else:
            # Use GET request
            postman_url = self._build_readable_url(endpoint, params)
            
            try:
                response = requests.get(
                    f"{self.base_url}{endpoint}",
                    headers=self.headers,
                    params=params,
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return {
                        "success": True,
                        "tag_filter": tag_filter or "all_tags",
                        "audience_ids": audience_ids,
                        "request_method": "GET",
                        "params_used": params,
                        "results": data.get("results", {}),
                        "raw_response": data,
                        "postman_url": postman_url,
                        "headers_needed": {"X-Api-Key": "YOUR_API_KEY"}
                    }
                else:
                    return {
                        "success": False,
                        "tag_filter": tag_filter or "all_tags",
                        "audience_ids": audience_ids,
                        "request_method": "GET",
                        "params_used": params,
                        "error": f"API Error {response.status_code}: {response.text}",
                        "status_code": response.status_code,
                        "postman_url": postman_url,
                        "headers_needed": {"X-Api-Key": "YOUR_API_KEY"}
                    }
            except Exception as e:
                return {
                    "success": False,
                    "tag_filter": tag_filter or "all_tags",
                    "audience_ids": audience_ids,
                    "request_method": "GET",
                    "error": f"GET request failed: {str(e)}",
                    "postman_url": postman_url,
                    "headers_needed": {"X-Api-Key": "YOUR_API_KEY"}
                }
    
    def get_demographics_analysis(
        self,
        entity_ids: Optional[List[str]] = None,
        tag_ids: Optional[List[str]] = None,
        signals: Optional[QlooSignals] = None,
        limit: int = 20
    ) -> Dict[str, Any]:
        """
        Get demographics analysis for given entities or tags
        
        Args:
            entity_ids: List of entity IDs to analyze demographics for
            tag_ids: List of tag IDs to analyze demographics for  
            signals: Additional signals to apply (demographics, location, entity queries)
            limit: Maximum number of results to return
            
        Returns:
            Dictionary with demographics analysis data and Postman URL
        """
        
        # At least one of entity_ids or tag_ids is required
        if not entity_ids and not tag_ids and not (signals and (signals.entity_ids or signals.tag_ids or signals.entity_queries)):
            raise ValueError("At least one of entity_ids, tag_ids, or signals with entities/tags is required")
        
        # Build query parameters
        params = {
            "filter.type": "urn:demographics",
            "take": str(limit)
        }
        
        # Add entity IDs
        if entity_ids:
            params["signal.interests.entities"] = ",".join(entity_ids)
            
        # Add tag IDs  
        if tag_ids:
            params["signal.interests.tags"] = ",".join(tag_ids)
        
        # Add additional signals if provided
        self._add_signal_params(params, signals)
        
        # Check if we need POST request
        use_post = self._should_use_post(signals)
        endpoint = "/v2/insights"
        
        if use_post:
            # Use POST request with JSON body
            body = self._build_post_body(params, signals)
            
            try:
                response = requests.post(
                    f"{self.base_url}{endpoint}",
                    headers=self.headers,
                    json=body,
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return {
                        "success": True,
                        "analysis_type": "demographics",
                        "entity_ids": entity_ids or [],
                        "tag_ids": tag_ids or [],
                        "request_method": "POST",
                        "params_used": body,
                        "results": data.get("results", {}),
                        "query_info": data.get("query", {}),
                        "raw_response": data,
                        "postman_url": f"{self.base_url}{endpoint} (POST)",
                        "postman_body": body,
                        "headers_needed": {"X-Api-Key": "YOUR_API_KEY"}
                    }
                else:
                    return {
                        "success": False,
                        "analysis_type": "demographics",
                        "entity_ids": entity_ids or [],
                        "tag_ids": tag_ids or [],
                        "request_method": "POST",
                        "params_used": body,
                        "error": f"API Error {response.status_code}: {response.text}",
                        "status_code": response.status_code,
                        "postman_url": f"{self.base_url}{endpoint} (POST)",
                        "postman_body": body,
                        "headers_needed": {"X-Api-Key": "YOUR_API_KEY"}
                    }
            except Exception as e:
                return {
                    "success": False,
                    "analysis_type": "demographics",
                    "entity_ids": entity_ids or [],
                    "tag_ids": tag_ids or [],
                    "request_method": "POST",
                    "error": f"POST request failed: {str(e)}",
                    "postman_url": f"{self.base_url}{endpoint} (POST)",
                    "postman_body": body,
                    "headers_needed": {"X-Api-Key": "YOUR_API_KEY"}
                }
        else:
            # Use GET request
            postman_url = self._build_readable_url(endpoint, params)
            
            try:
                response = requests.get(
                    f"{self.base_url}{endpoint}",
                    headers=self.headers,
                    params=params,
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return {
                        "success": True,
                        "analysis_type": "demographics",
                        "entity_ids": entity_ids or [],
                        "tag_ids": tag_ids or [],
                        "request_method": "GET",
                        "params_used": params,
                        "results": data.get("results", {}),
                        "raw_response": data,
                        "postman_url": postman_url,
                        "headers_needed": {"X-Api-Key": "YOUR_API_KEY"}
                    }
                else:
                    return {
                        "success": False,
                        "analysis_type": "demographics",
                        "entity_ids": entity_ids or [],
                        "tag_ids": tag_ids or [],
                        "request_method": "GET",
                        "params_used": params,
                        "error": f"API Error {response.status_code}: {response.text}",
                        "status_code": response.status_code,
                        "postman_url": postman_url,
                        "headers_needed": {"X-Api-Key": "YOUR_API_KEY"}
                    }
            except Exception as e:
                return {
                    "success": False,
                    "analysis_type": "demographics",
                    "entity_ids": entity_ids or [],
                    "tag_ids": tag_ids or [],
                    "request_method": "GET",
                    "error": f"GET request failed: {str(e)}",
                    "postman_url": postman_url,
                    "headers_needed": {"X-Api-Key": "YOUR_API_KEY"}
                }
    
    def get_heatmap_analysis(
        self,
        location_query: Optional[str] = None,
        location_filter: Optional[str] = None,
        entity_ids: Optional[List[str]] = None,
        tag_ids: Optional[List[str]] = None,
        signals: Optional[QlooSignals] = None,
        bias_trends: Optional[str] = None,
        boundary: Optional[str] = None,
        audience_weight: Optional[float] = None,
        limit: int = 20
    ) -> Dict[str, Any]:
        """
        Get heatmap analysis for given location and entities/tags
        
        Args:
            location_query: Location query string (e.g., "Toronto", "New York")
            location_filter: Location filter 
            entity_ids: List of entity IDs for heatmap analysis
            tag_ids: List of tag IDs for heatmap analysis
            signals: Additional signals to apply (demographics, location, entity queries)
            bias_trends: Optional bias trends parameter
            boundary: Optional heatmap boundary parameter
            audience_weight: Optional audience weight parameter
            limit: Maximum number of results to return
            
        Returns:
            Dictionary with heatmap analysis data and Postman URL
        """
        
        # At least one location parameter is required
        if not location_query and not location_filter and not (signals and signals.location):
            raise ValueError("At least one of location_query, location_filter, or signals.location is required")
        
        # Build query parameters
        params = {
            "filter.type": "urn:heatmap",
            "take": str(limit)
        }
        
        # Add location parameters
        if location_query:
            params["filter.location.query"] = location_query
        if location_filter:
            params["filter.location"] = location_filter
            
        # Add entity IDs
        if entity_ids:
            params["signal.interests.entities"] = ",".join(entity_ids)
            
        # Add tag IDs
        if tag_ids:
            params["signal.interests.tags"] = ",".join(tag_ids)
            
        # Add optional parameters
        if bias_trends:
            params["bias.trends"] = bias_trends
        if boundary:
            params["output.heatmap.boundary"] = boundary
        if audience_weight is not None:
            params["signal.demographics.audiences.weight"] = str(audience_weight)
        
        # Add additional signals if provided
        self._add_signal_params(params, signals)
        
        # Check if we need POST request
        use_post = self._should_use_post(signals)
        endpoint = "/v2/insights"
        
        if use_post:
            # Use POST request with JSON body
            body = self._build_post_body(params, signals)
            
            try:
                response = requests.post(
                    f"{self.base_url}{endpoint}",
                    headers=self.headers,
                    json=body,
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return {
                        "success": True,
                        "analysis_type": "heatmap",
                        "location_query": location_query,
                        "location_filter": location_filter,
                        "entity_ids": entity_ids or [],
                        "tag_ids": tag_ids or [],
                        "request_method": "POST",
                        "params_used": body,
                        "results": data.get("results", {}),
                        "query_info": data.get("query", {}),
                        "raw_response": data,
                        "postman_url": f"{self.base_url}{endpoint} (POST)",
                        "postman_body": body,
                        "headers_needed": {"X-Api-Key": "YOUR_API_KEY"}
                    }
                else:
                    return {
                        "success": False,
                        "analysis_type": "heatmap",
                        "location_query": location_query,
                        "location_filter": location_filter,
                        "entity_ids": entity_ids or [],
                        "tag_ids": tag_ids or [],
                        "request_method": "POST",
                        "params_used": body,
                        "error": f"API Error {response.status_code}: {response.text}",
                        "status_code": response.status_code,
                        "postman_url": f"{self.base_url}{endpoint} (POST)",
                        "postman_body": body,
                        "headers_needed": {"X-Api-Key": "YOUR_API_KEY"}
                    }
            except Exception as e:
                return {
                    "success": False,
                    "analysis_type": "heatmap",
                    "location_query": location_query,
                    "location_filter": location_filter,
                    "entity_ids": entity_ids or [],
                    "tag_ids": tag_ids or [],
                    "request_method": "POST",
                    "error": f"POST request failed: {str(e)}",
                    "postman_url": f"{self.base_url}{endpoint} (POST)",
                    "postman_body": body,
                    "headers_needed": {"X-Api-Key": "YOUR_API_KEY"}
                }
        else:
            # Use GET request
            postman_url = self._build_readable_url(endpoint, params)
            
            try:
                response = requests.get(
                    f"{self.base_url}{endpoint}",
                    headers=self.headers,
                    params=params,
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return {
                        "success": True,
                        "analysis_type": "heatmap",
                        "location_query": location_query,
                        "location_filter": location_filter,
                        "entity_ids": entity_ids or [],
                        "tag_ids": tag_ids or [],
                        "request_method": "GET",
                        "params_used": params,
                        "results": data.get("results", {}),
                        "raw_response": data,
                        "postman_url": postman_url,
                        "headers_needed": {"X-Api-Key": "YOUR_API_KEY"}
                    }
                else:
                    return {
                        "success": False,
                        "analysis_type": "heatmap",
                        "location_query": location_query,
                        "location_filter": location_filter,
                        "entity_ids": entity_ids or [],
                        "tag_ids": tag_ids or [],
                        "request_method": "GET",
                        "params_used": params,
                        "error": f"API Error {response.status_code}: {response.text}",
                        "status_code": response.status_code,
                        "postman_url": postman_url,
                        "headers_needed": {"X-Api-Key": "YOUR_API_KEY"}
                    }
            except Exception as e:
                return {
                    "success": False,
                    "analysis_type": "heatmap",
                    "location_query": location_query,
                    "location_filter": location_filter,
                    "entity_ids": entity_ids or [],
                    "tag_ids": tag_ids or [],
                    "request_method": "GET",
                    "error": f"GET request failed: {str(e)}",
                    "postman_url": postman_url,
                    "headers_needed": {"X-Api-Key": "YOUR_API_KEY"}
                }
    
    def get_multi_entity_insights(
        self,
        audience_ids: List[str],
        entity_types: List[str] = ["brands", "artists", "places"],
        signals: Optional[QlooSignals] = None,
        limit: int = 15
    ) -> Dict[str, Any]:
        """
        Get multiple types of entity insights for audience IDs (excludes tags)
        
        Args:
            audience_ids: List of audience IDs
            entity_types: List of entity types to get insights for
            signals: Additional signals to apply
            limit: Maximum results per entity type
            
        Returns:
            Dictionary with insights for each entity type plus all Postman URLs
        """
        results = {}
        postman_urls = {}
        
        for entity_type in entity_types:
            print(f"Getting {entity_type} insights...")
            insights = self.get_entity_insights(audience_ids, entity_type, signals, limit)
            results[entity_type] = insights
            
            if insights.get("request_method") == "POST":
                postman_urls[entity_type] = {
                    "url": insights.get("postman_url", ""),
                    "method": "POST",
                    "body": insights.get("postman_body", {})
                }
            else:
                postman_urls[entity_type] = insights.get("postman_url", "")
        
        return {
            "audience_ids": audience_ids,
            "entity_types": entity_types,
            "insights": results,
            "success": all(results[et].get("success", False) for et in entity_types),
            "postman_urls": postman_urls,
            "headers_needed": {"X-Api-Key": "YOUR_API_KEY"}
        }
    
    def print_postman_instructions(self, response_data: Dict[str, Any]):
        """
        Print instructions for testing in Postman
        """
        print("\n" + "="*60)
        print("POSTMAN TESTING INSTRUCTIONS")
        print("="*60)
        
        if "postman_body" in response_data:
            # POST request
            print(f"URL: {response_data.get('postman_url', 'N/A')}")
            print(f"Method: POST")
            print(f"Headers:")
            for key, value in response_data.get("headers_needed", {}).items():
                print(f"  {key}: {value}")
            print(f"Body (JSON):")
            print(json.dumps(response_data["postman_body"], indent=2))
                
        elif "postman_url" in response_data:
            # GET request
            print(f"URL: {response_data['postman_url']}")
            print(f"Method: GET")
            print(f"Headers:")
            for key, value in response_data.get("headers_needed", {}).items():
                print(f"  {key}: {value}")
                
        elif "postman_urls" in response_data:
            print("Multiple URLs available:")
            for name, url_info in response_data["postman_urls"].items():
                print(f"\n{name}:")
                if isinstance(url_info, dict):
                    print(f"  URL: {url_info.get('url', 'N/A')}")
                    print(f"  Method: {url_info.get('method', 'GET')}")
                    if 'body' in url_info:
                        print(f"  Body (JSON):")
                        print(f"    {json.dumps(url_info['body'], indent=4)}")
                else:
                    print(f"  URL: {url_info}")
                    print(f"  Method: GET")
                print(f"  Headers:")
                for key, value in response_data.get("headers_needed", {}).items():
                    print(f"    {key}: {value}")
        
        print("\n" + "="*60)

    # Legacy method for backward compatibility
    def get_insights(self, audience_ids: List[str], entity_type: str = "brands", signals: Optional[QlooSignals] = None, limit: int = 20) -> Dict[str, Any]:
        """
        Legacy method - use get_entity_insights() or get_tag_insights() instead
        """
        if entity_type == "tags":
            return self.get_tag_insights(audience_ids, signals, limit)
        else:
            return self.get_entity_insights(audience_ids, entity_type, signals, limit)
    
    def get_multi_insights(self, audience_ids: List[str], entity_types: List[str] = ["brands", "artists"], signals: Optional[QlooSignals] = None, limit: int = 15) -> Dict[str, Any]:
        """
        Legacy method - use get_multi_entity_insights() or get_tag_insights() separately
        """
        return self.get_multi_entity_insights(audience_ids, entity_types, signals, limit)
    

    def search_tags(
        self,
        query: Optional[str] = None,
        tag_types: Optional[List[str]] = None,
        parent_types: Optional[List[str]] = None,
        signals: Optional[QlooSignals] = None,
        limit: int = 20
    ) -> Dict[str, Any]:
        """
        Search for tags by name/query with optional filters
        
        Args:
            query: Text to search for in tag names (partial match)
            tag_types: List of tag types to filter by (e.g., ["urn:tag:genre:restaurant"])
            parent_types: List of parent entity types (e.g., ["urn:entity:movie"])
            signals: Additional signals for context
            limit: Maximum number of results to return
            
        Returns:
            Dictionary with tag search results
        """
        
        # Build query parameters
        params = {
            "take": str(limit)
        }
        
        # Add query filter if provided
        if query:
            params["filter.query"] = query.strip()
        
        # Add tag type filters
        if tag_types:
            params["filter.tag.types"] = ",".join(tag_types)
        
        # Add parent type filters
        if parent_types:
            params["filter.parents.types"] = ",".join(parent_types)
        
        # Add signals if provided
        self._add_signal_params(params, signals)
        
        endpoint = "/v2/tags"
        postman_url = self._build_readable_url(endpoint, params)
        
        try:
            response = requests.get(
                f"{self.base_url}{endpoint}",
                headers=self.headers,
                params=params,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                tags = data.get("results", {}).get("tags", [])
                
                return {
                    "success": True,
                    "query": query or "all_tags",
                    "tag_types": tag_types or [],
                    "parent_types": parent_types or [],
                    "tags": tags,
                    "total_found": len(tags),
                    "raw_response": data,
                    "postman_url": postman_url,
                    "headers_needed": {"X-Api-Key": "YOUR_API_KEY"}
                }
            else:
                return {
                    "success": False,
                    "query": query or "all_tags",
                    "tag_types": tag_types or [],
                    "parent_types": parent_types or [],
                    "tags": [],
                    "total_found": 0,
                    "error": f"API Error {response.status_code}: {response.text}",
                    "status_code": response.status_code,
                    "postman_url": postman_url,
                    "headers_needed": {"X-Api-Key": "YOUR_API_KEY"}
                }
                
        except Exception as e:
            return {
                "success": False,
                "query": query or "all_tags",
                "tag_types": tag_types or [],
                "parent_types": parent_types or [],
                "tags": [],
                "total_found": 0,
                "error": f"Tag search failed: {str(e)}",
                "postman_url": postman_url,
                "headers_needed": {"X-Api-Key": "YOUR_API_KEY"}
            }

    def get_tag_types(self, parent_types: Optional[List[str]] = None, limit: int = 50) -> Dict[str, Any]:
        """
        Get available tag types, optionally filtered by parent entity types
        
        Args:
            parent_types: List of parent entity types to filter by
            limit: Maximum number of results to return
            
        Returns:
            Dictionary with tag types data
        """
        
        params = {
            "take": str(limit)
        }
        
        if parent_types:
            params["filter.parents.types"] = ",".join(parent_types)
        
        endpoint = "/v2/tags/types"
        postman_url = self._build_readable_url(endpoint, params)
        
        try:
            response = requests.get(
                f"{self.base_url}{endpoint}",
                headers=self.headers,
                params=params,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                entities = data.get("results", {}).get("entities", [])
                
                return {
                    "success": True,
                    "parent_types": parent_types or [],
                    "tag_types": entities,
                    "total_found": len(entities),
                    "raw_response": data,
                    "postman_url": postman_url,
                    "headers_needed": {"X-Api-Key": "YOUR_API_KEY"}
                }
            else:
                return {
                    "success": False,
                    "parent_types": parent_types or [],
                    "tag_types": [],
                    "total_found": 0,
                    "error": f"API Error {response.status_code}: {response.text}",
                    "status_code": response.status_code,
                    "postman_url": postman_url,
                    "headers_needed": {"X-Api-Key": "YOUR_API_KEY"}
                }
                
        except Exception as e:
            return {
                "success": False,
                "parent_types": parent_types or [],
                "tag_types": [],
                "total_found": 0,
                "error": f"Tag types request failed: {str(e)}",
                "postman_url": postman_url,
                "headers_needed": {"X-Api-Key": "YOUR_API_KEY"}
            }

    def get_audience_types(self, limit: int = 50) -> Dict[str, Any]:
        """
        Get available audience types
        
        Args:
            limit: Maximum number of results to return
            
        Returns:
            Dictionary with audience types data
        """
        
        params = {
            "take": str(limit)
        }
        
        endpoint = "/v2/audiences/types"
        postman_url = self._build_readable_url(endpoint, params)
        
        try:
            response = requests.get(
                f"{self.base_url}{endpoint}",
                headers=self.headers,
                params=params,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                
                return {
                    "success": True,
                    "audience_types": data.get("results", {}),
                    "raw_response": data,
                    "postman_url": postman_url,
                    "headers_needed": {"X-Api-Key": "YOUR_API_KEY"}
                }
            else:
                return {
                    "success": False,
                    "audience_types": {},
                    "error": f"API Error {response.status_code}: {response.text}",
                    "status_code": response.status_code,
                    "postman_url": postman_url,
                    "headers_needed": {"X-Api-Key": "YOUR_API_KEY"}
                }
                
        except Exception as e:
            return {
                "success": False,
                "audience_types": {},
                "error": f"Audience types request failed: {str(e)}",
                "postman_url": postman_url,
                "headers_needed": {"X-Api-Key": "YOUR_API_KEY"}
            }

    def find_entity_by_name(self, entity_name: str, entity_type: Optional[str] = None, get_best_match: bool = True) -> Dict[str, Any]:
        """
        Find entity ID by name (simplified wrapper around search_entities)
        
        Args:
            entity_name: Name to search for
            entity_type: Optional entity type to filter by
            get_best_match: If True, returns only the best match; if False, returns top 5
            
        Returns:
            Dictionary with entity ID and details
        """
        
        entity_types = [entity_type] if entity_type else None
        limit = 1 if get_best_match else 5
        
        result = self.search_entities(
            query=entity_name,
            entity_types=entity_types,
            limit=limit,
            sort_by="match"
        )
        
        if result.get("success") and result.get("entities"):
            if get_best_match:
                best_entity = result["entities"][0]
                return {
                    "success": True,
                    "entity_name": entity_name,
                    "entity_id": best_entity.id,
                    "entity_type": best_entity.entity_type,
                    "match_name": best_entity.name,
                    "affinity_score": best_entity.affinity_score,
                    "entity_details": best_entity
                }
            else:
                return {
                    "success": True,
                    "entity_name": entity_name,
                    "entities": [
                        {
                            "entity_id": entity.id,
                            "name": entity.name,
                            "entity_type": entity.entity_type,
                            "affinity_score": entity.affinity_score
                        } for entity in result["entities"]
                    ],
                    "total_found": len(result["entities"])
                }
        else:
            return {
                "success": False,
                "entity_name": entity_name,
                "entity_id": None,
                "error": result.get("error", "No entities found"),
                "entities": []
            }

    def find_tags_by_name(self, tag_name: str, tag_type: Optional[str] = None, limit: int = 5) -> Dict[str, Any]:
        """
        Find tags by name (simplified wrapper around search_tags)
        
        Args:
            tag_name: Name to search for
            tag_type: Optional tag type to filter by
            limit: Maximum number of results
            
        Returns:
            Dictionary with tag IDs and details
        """
        
        tag_types = [tag_type] if tag_type else None
        
        result = self.search_tags(
            query=tag_name,
            tag_types=tag_types,
            limit=limit
        )
        
        if result.get("success") and result.get("tags"):
            return {
                "success": True,
                "tag_name": tag_name,
                "tags": [
                    {
                        "tag_id": tag.get("tag_id"),
                        "name": tag.get("name"),
                        "subtype": tag.get("subtype"),
                        "types": tag.get("types", [])
                    } for tag in result["tags"]
                ],
                "total_found": len(result["tags"])
            }
        else:
            return {
                "success": False,
                "tag_name": tag_name,
                "tags": [],
                "error": result.get("error", "No tags found")
            }


        # # Add this to qloo.py
        # def search_audiences(self, signals: QlooSignals, limit: int = 50) -> Dict[str, Any]:
        #     """
        #     Simplified audience search for agent pipeline
        #     Returns audiences that match the provided signals
        #     """
        #     try:
        #         result = self.find_audiences(signals=signals, limit=limit)
                
        #         if result.get("total_found", 0) > 0:
        #             audiences = result.get("audiences", [])
        #             return {
        #                 "success": True,
        #                 "audiences": [
        #                     {
        #                         "id": aud.id,
        #                         "name": aud.name,
        #                         "parent_type": aud.parent_type
        #                     } for aud in audiences
        #                 ],
        #                 "total_found": len(audiences),
        #                 "message": f"Found {len(audiences)} relevant audiences"
        #             }
        #         else:
        #             return {
        #                 "success": False,
        #                 "error": "No audiences found for the provided signals",
        #                 "audiences": [],
        #                 "total_found": 0
        #             }
                    
        #     except Exception as e:
        #         return {
        #             "success": False,
        #             "error": f"Audience search failed: {str(e)}",
        #             "audiences": [],
        #             "total_found": 0
        #         }
            


