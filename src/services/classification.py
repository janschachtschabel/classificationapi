"""Classification service for text analysis using OpenAI and SKOS vocabularies."""

import json
import time
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import httpx
from cachetools import TTLCache
from openai import AsyncOpenAI
from pydantic import HttpUrl
from sentence_transformers import SentenceTransformer

from ..api.errors import OpenAIError, VocabularyFetchError
from ..core.config import settings
from ..core.logging import logger
from ..schemas.classification import (
    ClassificationMatch,
    ClassificationMetadata,
    ClassificationMode,
    ClassificationRequest,
    ClassificationResponse,
    ClassificationResult,
    DescriptiveFields,
    ResourceSuggestionFields,
    ResourceSuggestionFilterSuggestion,
)


class SKOSProcessingStats:
    """Track SKOS processing statistics."""

    def __init__(self) -> None:
        self.concepts_found = 0
        self.concepts_processed = 0
        self.semantic_filtering_used = False
        self.semantic_filtering_disabled = False
        self.max_concepts_per_vocabulary = 0
        self.vocabularies_filtered = 0
        self.reduction_percent = 0.0
        # Block processing attributes
        self.blocks_count = 0
        self.block_size = 0
        self.block_processing_used = False
        self.candidates_per_block = 0
        self.final_refinement_used = False


class ClassificationService:
    """Service for handling text classification requests."""

    def __init__(self) -> None:
        """Initialize the classification service."""
        self.openai_client = AsyncOpenAI(api_key=settings.openai_api_key, timeout=settings.openai_timeout_seconds)
        self.vocabulary_cache: TTLCache[str, dict[str, Any]] = TTLCache(
            maxsize=settings.cache_max_size, ttl=settings.cache_ttl_seconds
        )
        self.http_client = httpx.AsyncClient(timeout=settings.http_timeout_seconds)

        # Cross-encoder for semantic similarity
        self.cross_encoder: SentenceTransformer | None = None
        self._initialize_cross_encoder()

    async def classify_text(self, request: ClassificationRequest) -> ClassificationResponse:
        """
        Classify text based on the provided request.

        Args:
            request: Classification request containing text and configuration

        Returns:
            ClassificationResponse: Classification results

        Raises:
            VocabularyFetchError: If vocabulary fetching fails
            OpenAIError: If OpenAI API call fails
        """
        start_time = datetime.now(UTC)
        classification_id = uuid4()

        # Initialize SKOS processing stats
        skos_stats = SKOSProcessingStats()

        logger.info(
            f"Starting classification {classification_id}",
            extra={"classification_id": str(classification_id), "mode": request.mode, "text_length": len(request.text)},
        )

        try:
            # Initialize results, descriptive fields, and resource suggestions
            results = []
            descriptive_fields = None
            resource_suggestion_fields = None

            # Prepare vocabularies/categories if mode is specified
            vocabularies = []
            if request.mode == ClassificationMode.SKOS:
                if request.vocabulary_sources and len([url for url in request.vocabulary_sources if url]) > 0:
                    valid_urls = [HttpUrl(url) for url in request.vocabulary_sources if url]
                    vocabularies = await self._fetch_vocabularies(valid_urls)
                    # Collect SKOS stats
                    for vocab in vocabularies:
                        skos_stats.concepts_found += len(vocab.get("values", []))
            elif request.mode == ClassificationMode.CUSTOM:
                if request.custom_categories and len(request.custom_categories) > 0:
                    vocabularies = self._prepare_custom_categories(request.custom_categories)

            # Get model settings
            model = request.model or settings.openai_default_model
            temperature = request.temperature or settings.default_temperature
            max_tokens = request.max_tokens or settings.default_max_tokens

            # Perform classification if vocabularies are provided
            if vocabularies:
                results = await self._classify_with_openai(
                    text=request.text,
                    vocabularies=vocabularies,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    skos_stats=skos_stats,
                    enable_semantic_filtering=request.enable_semantic_filtering,
                    max_concepts_per_vocabulary=request.max_concepts_per_vocabulary,
                )

            # Generate descriptive fields if requested
            if request.generate_descriptive_fields:
                descriptive_fields = await self._generate_descriptive_fields(
                    text=request.text, model=model, temperature=temperature, max_tokens=max_tokens
                )

            # Generate resource suggestions if requested
            if request.resource_suggestion:
                # For resource suggestions, use vocabularies from classification or handle empty list
                resource_vocabularies = vocabularies

                # Only use default vocabularies if vocabulary_sources is None (not explicitly empty)
                if (
                    not resource_vocabularies
                    and request.vocabulary_sources is None
                    and request.mode != ClassificationMode.CUSTOM
                ):
                    # Use default SKOS vocabularies for resource suggestions
                    default_vocab_urls = [
                        "https://vocabs.openeduhub.de/w3id.org/openeduhub/vocabs/discipline/",
                        "https://vocabs.openeduhub.de/w3id.org/openeduhub/vocabs/educationalContext/",
                        "https://vocabs.openeduhub.de/w3id.org/openeduhub/vocabs/new_lrt/",
                    ]
                    default_vocab_http_urls = [HttpUrl(url) for url in default_vocab_urls]
                    resource_vocabularies = await self._fetch_vocabularies(default_vocab_http_urls)
                # If vocabulary_sources is explicitly empty list [], use empty vocabularies

                resource_suggestion_fields = await self._generate_resource_suggestions(
                    text=request.text,
                    vocabularies=resource_vocabularies,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

            # Calculate processing time
            processing_time_ms = int((datetime.now(UTC) - start_time).total_seconds() * 1000)

            # Build metadata with SKOS processing info
            metadata = ClassificationMetadata(
                model=model,
                model_settings={"temperature": temperature, "max_tokens": max_tokens},
                timestamp=datetime.now(UTC),
                processing_time_ms=processing_time_ms,
                skos_concepts_found=None,
                skos_concepts_processed=None,
                semantic_filtering_used=None,
                semantic_filtering_disabled=None,
                semantic_filtering_info=None,
            )

            # Add SKOS processing stats if in SKOS mode
            if request.mode == ClassificationMode.SKOS:
                metadata.skos_concepts_found = skos_stats.concepts_found
                metadata.skos_concepts_processed = skos_stats.concepts_processed
                metadata.semantic_filtering_used = skos_stats.semantic_filtering_used
                metadata.semantic_filtering_disabled = skos_stats.semantic_filtering_disabled
                if skos_stats.semantic_filtering_used:
                    metadata.semantic_filtering_info = {
                        "max_concepts_per_vocabulary": skos_stats.max_concepts_per_vocabulary,
                        "vocabularies_filtered": skos_stats.vocabularies_filtered,
                        "reduction_percent": skos_stats.reduction_percent,
                    }

            # Create response
            response = ClassificationResponse(
                classification_id=classification_id,
                status="completed",
                results=results,
                metadata=metadata,
                descriptive_fields=descriptive_fields,
                resource_suggestion_fields=resource_suggestion_fields,
            )

            logger.info(
                f"Classification {classification_id} completed",
                extra={
                    "classification_id": str(classification_id),
                    "processing_time_ms": processing_time_ms,
                    "results_count": len(results),
                },
            )

            return response

        except Exception as e:
            logger.error(
                f"Classification {classification_id} failed: {str(e)}",
                extra={"classification_id": str(classification_id), "error": str(e)},
            )
            raise

    def _initialize_cross_encoder(self) -> None:
        """
        Initialize the cross-encoder model for semantic similarity.
        """
        try:
            # Use a lightweight multilingual model for German/English content
            model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            self.cross_encoder = SentenceTransformer(model_name)
            logger.info(f"Cross-encoder initialized: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize cross-encoder: {str(e)}")
            self.cross_encoder = None

    def _find_relevant_concepts(
        self, query_text: str, vocabularies: list[dict[str, Any]], top_k: int = 100
    ) -> list[dict[str, Any]]:
        """
        Use cross-encoder to find the most relevant concepts for the query.

        Args:
            query_text: The input text to find relevant concepts for
            vocabularies: All loaded vocabularies with complete concept sets
            top_k: Number of most relevant concepts to return per vocabulary

        Returns:
            Filtered vocabularies with only the most relevant concepts
        """
        if not self.cross_encoder:
            logger.warning("Cross-encoder not available, returning original vocabularies")
            return vocabularies

        filtered_vocabularies = []
        total_original_concepts = 0
        total_filtered_concepts = 0

        for vocab_data in vocabularies:
            vocab_name = vocab_data.get("property", "Unknown")
            concepts = vocab_data.get("values", [])
            concept_count = len(concepts)
            total_original_concepts += concept_count

            logger.info(
                f"Processing vocabulary '{vocab_name}' with {concept_count} concepts",
                extra={"vocab_name": vocab_name, "concept_count": concept_count},
            )

            # Apply filtering only if concept count exceeds the limit
            if concept_count <= top_k:
                logger.info(
                    f"Vocabulary '{vocab_name}' has {concept_count} concepts (≤{top_k}), no filtering applied",
                    extra={"vocab_name": vocab_name, "concept_count": concept_count, "limit": top_k},
                )
                # Still preserve original values for consistency
                vocab_copy = vocab_data.copy()
                vocab_copy["original_values"] = concepts
                filtered_vocabularies.append(vocab_copy)
                total_filtered_concepts += concept_count
                continue

            try:
                # Extract concept labels for similarity comparison
                concept_labels = []
                for concept in concepts:
                    label = concept.get("label", concept.get("prefLabel", ""))
                    if label:
                        concept_labels.append(label)
                    else:
                        concept_labels.append(concept.get("id", "Unknown"))

                if not concept_labels:
                    logger.warning(f"No valid labels found in vocabulary '{vocab_name}'")
                    filtered_vocabularies.append(vocab_data)
                    total_filtered_concepts += concept_count
                    continue

                # Calculate semantic similarity scores
                logger.info(f"Computing semantic similarity for {len(concept_labels)} concepts...")

                # Create pairs for cross-encoder
                pairs = [[query_text, label] for label in concept_labels]

                # Get similarity scores
                scores = self.cross_encoder.predict(pairs)  # type: ignore[operator]
                # Convert to list if it's a tensor/array
                if hasattr(scores, "tolist"):
                    scores = scores.tolist()
                elif hasattr(scores, "__iter__"):
                    scores = list(scores)

                # Get top-k most relevant concepts
                top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

                # Filter concepts based on top scores
                filtered_concepts = [concepts[i] for i in top_indices]

                # Create filtered vocabulary and preserve original count for statistics
                filtered_vocab = vocab_data.copy()
                filtered_vocab["original_values"] = concepts  # Store original for statistics
                filtered_vocab["values"] = filtered_concepts
                filtered_vocabularies.append(filtered_vocab)

                total_filtered_concepts += len(filtered_concepts)

                logger.info(
                    f"Vocabulary '{vocab_name}' filtered: {concept_count} → {len(filtered_concepts)} concepts",
                    extra={
                        "vocab_name": vocab_name,
                        "original_count": concept_count,
                        "filtered_count": len(filtered_concepts),
                        "reduction_percent": round((1 - len(filtered_concepts) / concept_count) * 100, 1),
                    },
                )

            except Exception as e:
                logger.error(
                    f"Error in semantic search for vocabulary '{vocab_name}': {str(e)}",
                    extra={"vocab_name": vocab_name, "error": str(e)},
                )
                # Fallback: use original vocabulary
                filtered_vocabularies.append(vocab_data)
                total_filtered_concepts += concept_count

        logger.info(
            f"Semantic filtering complete: {total_original_concepts} → {total_filtered_concepts} concepts",
            extra={
                "total_original_concepts": total_original_concepts,
                "total_filtered_concepts": total_filtered_concepts,
                "reduction_percent": round((1 - total_filtered_concepts / total_original_concepts) * 100, 1)
                if total_original_concepts > 0
                else 0,
            },
        )

        return filtered_vocabularies

    async def _fetch_vocabularies(self, urls: list[HttpUrl]) -> list[dict[str, Any]]:
        """
        Fetch and parse SKOS vocabularies from URLs.

        Args:
            urls: List of vocabulary URLs

        Returns:
            List of parsed vocabulary dictionaries

        Raises:
            VocabularyFetchError: If fetching fails
        """
        vocabularies = []

        for url in urls:
            url_str = str(url)

            # Check cache first
            if url_str in self.vocabulary_cache:
                logger.debug(f"Using cached vocabulary from {url_str}")
                vocabularies.append(self.vocabulary_cache[url_str])
                continue

            try:
                logger.info(f"Fetching vocabulary from {url_str}")
                start_time = time.time()

                # Use longer timeout for large vocabulary files
                timeout = httpx.Timeout(60.0, connect=10.0)  # 60s total, 10s connect

                response = await self.http_client.get(url_str, timeout=timeout)
                response.raise_for_status()

                fetch_time = time.time() - start_time
                content_length = len(response.content)

                logger.info(
                    f"Downloaded vocabulary in {fetch_time:.2f}s, size: {content_length:,} bytes",
                    extra={"url": url_str, "fetch_time_seconds": fetch_time, "content_size_bytes": content_length},
                )

                # Parse JSON with better error handling
                try:
                    vocab_data = response.json()
                except json.JSONDecodeError as e:
                    error_msg = f"Invalid JSON in vocabulary from {url_str}: {str(e)}"
                    logger.error(error_msg)
                    raise VocabularyFetchError(error_msg, {"url": url_str, "json_error": str(e)})

                # Validate basic structure
                if not isinstance(vocab_data, dict):
                    error_msg = f"Vocabulary from {url_str} is not a valid JSON object"
                    logger.error(error_msg)
                    raise VocabularyFetchError(error_msg, {"url": url_str})

                # Parse vocabulary with timing
                parse_start = time.time()
                parsed_vocab = self._parse_skos_vocabulary(vocab_data, url_str)
                parse_time = time.time() - parse_start

                # Cache the parsed vocabulary
                self.vocabulary_cache[url_str] = parsed_vocab
                vocabularies.append(parsed_vocab)

                logger.info(
                    f"Successfully processed vocabulary '{parsed_vocab['property']}' in {parse_time:.2f}s with {len(parsed_vocab['values'])} concepts",
                    extra={
                        "url": url_str,
                        "vocab_name": parsed_vocab["property"],
                        "values_count": len(parsed_vocab["values"]),
                        "parse_time_seconds": parse_time,
                        "total_time_seconds": fetch_time + parse_time,
                    },
                )

            except httpx.TimeoutException as e:
                error_msg = f"Timeout fetching vocabulary from {url_str}: {str(e)}"
                logger.error(error_msg, extra={"url": url_str, "timeout_seconds": 60})
                raise VocabularyFetchError(error_msg, {"url": url_str, "error_type": "timeout"})

            except httpx.HTTPStatusError as e:
                error_msg = f"HTTP error fetching vocabulary from {url_str}: {e.response.status_code} {e.response.reason_phrase}"
                logger.error(error_msg, extra={"url": url_str, "status_code": e.response.status_code})
                raise VocabularyFetchError(error_msg, {"url": url_str, "status_code": e.response.status_code})

            except httpx.HTTPError as e:
                error_msg = f"Network error fetching vocabulary from {url_str}: {str(e)}"
                logger.error(error_msg, extra={"url": url_str, "error_type": "network"})
                raise VocabularyFetchError(error_msg, {"url": url_str, "error_type": "network"})

            except VocabularyFetchError:
                # Re-raise vocabulary fetch errors as-is
                raise

            except Exception as e:
                error_msg = f"Unexpected error processing vocabulary from {url_str}: {str(e)}"
                logger.error(error_msg, extra={"url": url_str, "error_type": "unexpected", "error": str(e)})
                raise VocabularyFetchError(error_msg, {"url": url_str, "error_type": "unexpected"})

        total_concepts = sum(len(vocab.get("values", [])) for vocab in vocabularies)
        logger.info(
            f"Successfully loaded {len(vocabularies)} vocabularies with {total_concepts} total concepts",
            extra={"vocabularies_count": len(vocabularies), "total_concepts": total_concepts},
        )

        return vocabularies

    def _parse_skos_vocabulary(self, vocab_data: dict[str, Any], url: str) -> dict[str, Any]:
        """
        Parse SKOS vocabulary data with adaptive limits based on vocabulary size.

        Args:
            vocab_data: Raw vocabulary data from JSON
            url: Source URL

        Returns:
            Parsed vocabulary dictionary
        """
        vocab_name = url.split("/")[-1].replace(".json", "")

        # Extract property name from title
        property_name = ""
        if "title" in vocab_data:
            title = vocab_data["title"]
            if isinstance(title, dict):
                property_name = title.get("de", "") or title.get("en", "")
            else:
                property_name = str(title)

        if not property_name:
            property_name = vocab_name

        # Determine if this is a large vocabulary by checking top concepts count
        top_concepts = vocab_data.get("hasTopConcept", [])
        is_large_vocab = len(top_concepts) > 50  # Heuristic: >50 top concepts = large vocab

        # Set adaptive limits based on vocabulary size
        if is_large_vocab:
            max_concepts = settings.max_concepts_large  # 1000 for large vocabs
            max_depth = settings.max_depth_large  # 5 levels for large vocabs
            logger.info(
                f"Large vocabulary detected: using enhanced limits (concepts: {max_concepts}, depth: {max_depth})"
            )
        else:
            max_concepts = settings.max_concepts_standard  # 200 for standard vocabs
            max_depth = settings.max_depth_standard  # 3 levels for standard vocabs

        # Extract values with adaptive limits
        values: list[dict[str, str]] = []

        logger.info(
            f"Starting to parse vocabulary '{property_name}' from {url}",
            extra={
                "url": url,
                "vocab_name": vocab_name,
                "is_large_vocab": is_large_vocab,
                "max_concepts": max_concepts,
                "max_depth": max_depth,
                "top_concepts_count": len(top_concepts),
            },
        )

        try:
            if "hasTopConcept" in vocab_data:
                top_concepts = vocab_data.get("hasTopConcept", [])
                logger.info(
                    f"Found {len(top_concepts)} top-level concepts in vocabulary '{property_name}'",
                    extra={"url": url, "top_concepts_count": len(top_concepts)},
                )

                # Process top concepts with limits
                for i, item in enumerate(top_concepts):
                    if len(values) >= max_concepts:
                        logger.warning(
                            f"Vocabulary {property_name} has too many concepts. Limited to {max_concepts} for performance.",
                            extra={
                                "url": url,
                                "total_extracted": len(values),
                                "remaining_top_concepts": len(top_concepts) - i,
                            },
                        )
                        break

                    remaining_quota = max_concepts - len(values)
                    self._extract_concept_values(item, values, remaining_quota, current_depth=0, max_depth=max_depth)

                    # Log progress for large vocabularies
                    if i > 0 and i % 10 == 0:
                        logger.debug(
                            f"Processed {i + 1}/{len(top_concepts)} top concepts, extracted {len(values)} total concepts",
                            extra={"url": url, "progress": f"{i + 1}/{len(top_concepts)}"},
                        )

            logger.info(
                f"Successfully parsed vocabulary '{property_name}' with {len(values)} concepts",
                extra={"url": url, "vocab_name": vocab_name, "concepts_count": len(values)},
            )

        except Exception as e:
            logger.error(f"Error parsing vocabulary '{property_name}': {str(e)}", extra={"url": url, "error": str(e)})
            # Continue with whatever concepts we managed to extract
            logger.info(
                f"Partial vocabulary '{property_name}' extracted with {len(values)} concepts despite error",
                extra={"url": url, "concepts_count": len(values)},
            )

        return {"name": vocab_name, "property": property_name, "values": values}

    def _extract_concept_values(
        self,
        concept: dict[str, Any],
        values: list[dict[str, str]],
        max_remaining: int | None = None,
        current_depth: int = 0,
        max_depth: int = 3,
    ) -> None:
        """
        Recursively extract concept values from SKOS data with depth and quota limits.

        Args:
            concept: SKOS concept dictionary
            values: List to append extracted values to
            max_remaining: Maximum number of additional concepts to extract
            current_depth: Current recursion depth
            max_depth: Maximum allowed recursion depth
        """
        # Check limits
        if max_remaining is not None and max_remaining <= 0:
            return

        if current_depth >= max_depth:
            logger.debug(f"Reached maximum depth {max_depth}, stopping recursion")
            return

        # Validate concept structure
        if not isinstance(concept, dict):
            logger.warning(f"Invalid concept structure: expected dict, got {type(concept)}")
            return

        concept_id = concept.get("id", "")

        # Get preferred label with better error handling
        pref_label = ""
        try:
            if "prefLabel" in concept:
                label = concept["prefLabel"]
                if isinstance(label, dict):
                    pref_label = label.get("de", "") or label.get("en", "") or label.get("fr", "")
                    # If no language-specific label found, try to get any available value
                    if not pref_label and label:
                        pref_label = next(iter(label.values()), "") if isinstance(label, dict) else str(label)
                else:
                    pref_label = str(label)
        except Exception as e:
            logger.debug(f"Error extracting preferred label: {str(e)}")

        # Add alternative labels for better matching (limited to prevent bloat)
        alt_labels = []
        try:
            if "altLabel" in concept:
                alt_label = concept["altLabel"]
                if isinstance(alt_label, dict):
                    alt_labels_raw = alt_label.get("de", []) or alt_label.get("en", []) or alt_label.get("fr", [])
                    if isinstance(alt_labels_raw, list):
                        alt_labels = alt_labels_raw[:2]  # Reduced to 2 alternative labels
                    elif isinstance(alt_labels_raw, str):
                        alt_labels = [alt_labels_raw]
        except Exception as e:
            logger.debug(f"Error extracting alternative labels: {str(e)}")

        # Add concept if valid
        if concept_id and pref_label:
            concept_entry = {"id": concept_id, "label": pref_label}
            if alt_labels:
                concept_entry["alt_labels"] = alt_labels
            values.append(concept_entry)

            if max_remaining is not None:
                max_remaining -= 1

        # Process narrower concepts with stricter limits
        try:
            narrower_concepts = concept.get("narrower", [])
            if narrower_concepts and (max_remaining is None or max_remaining > 0):
                # Use configurable limit for narrower concepts
                max_narrower = min(len(narrower_concepts), settings.max_narrower_concepts)

                for narrower in narrower_concepts[:max_narrower]:
                    if max_remaining is not None and max_remaining <= 0:
                        break

                    remaining_after = max_remaining - 1 if max_remaining is not None else None
                    self._extract_concept_values(narrower, values, remaining_after, current_depth + 1, max_depth)

                    if max_remaining is not None:
                        max_remaining = remaining_after

        except Exception as e:
            logger.debug(f"Error processing narrower concepts: {str(e)}")

    def _prepare_custom_categories(self, categories: dict[str, list[str]]) -> list[dict[str, Any]]:
        """
        Prepare custom categories for classification.

        Args:
            categories: Custom categories dictionary

        Returns:
            List of vocabulary-like dictionaries
        """
        vocabularies = []

        for property_name, category_list in categories.items():
            values = []
            for i, category in enumerate(category_list):
                values.append({"id": f"custom_{property_name}_{i}", "label": category})

            vocabularies.append({"name": f"custom_{property_name}", "property": property_name, "values": values})

        return vocabularies

    async def _classify_with_openai(
        self,
        text: str,
        vocabularies: list[dict[str, Any]],
        model: str,
        temperature: float,
        max_tokens: int,
        skos_stats: SKOSProcessingStats,
        enable_semantic_filtering: bool = True,
        max_concepts_per_vocabulary: int = 100,
    ) -> list[ClassificationResult]:
        """
        Perform classification using OpenAI API with Cross-Encoder semantic filtering for large vocabularies.

        Args:
            text: Text to classify
            vocabularies: List of vocabulary dictionaries
            model: OpenAI model to use
            temperature: Temperature setting
            max_tokens: Maximum tokens
            skos_stats: SKOS processing statistics
            enable_semantic_filtering: If True, apply Cross-Encoder filtering to reduce concepts
            max_concepts_per_vocabulary: Maximum concepts per vocabulary when filtering is enabled

        Returns:
            List of classification results

        Raises:
            OpenAIError: If OpenAI API call fails
        """
        # Apply semantic filtering if enabled
        if enable_semantic_filtering:
            logger.info(
                f"Applying Cross-Encoder semantic filtering with max {max_concepts_per_vocabulary} concepts per vocabulary"
            )
            original_concept_count = sum(len(vocab.get("values", [])) for vocab in vocabularies)
            vocabularies = self._find_relevant_concepts(text, vocabularies, max_concepts_per_vocabulary)
            filtered_concept_count = sum(len(vocab.get("values", [])) for vocab in vocabularies)

            # Update SKOS stats
            skos_stats.semantic_filtering_used = True
            skos_stats.max_concepts_per_vocabulary = max_concepts_per_vocabulary

            # Count vocabularies that were actually filtered (reduced from original size)
            vocabularies_filtered = 0
            for v in vocabularies:
                # Get original count from original_values if available, otherwise this vocab wasn't filtered
                if "original_values" in v:
                    original_count = len(v["original_values"])
                    current_count = len(v.get("values", []))
                    if current_count < original_count:
                        vocabularies_filtered += 1
                        logger.debug(
                            f"Vocabulary '{v.get('property', 'Unknown')}' filtered: {original_count} → {current_count} concepts"
                        )
                else:
                    # No original_values means this vocabulary wasn't filtered
                    logger.debug(f"Vocabulary '{v.get('property', 'Unknown')}' not filtered (no original_values)")

            skos_stats.vocabularies_filtered = vocabularies_filtered
            skos_stats.reduction_percent = (
                round((1 - filtered_concept_count / original_concept_count) * 100, 1)
                if original_concept_count > 0
                else 0.0
            )

            logger.info(
                f"Semantic filtering stats: {vocabularies_filtered} vocabularies filtered, {skos_stats.reduction_percent}% reduction"
            )
        else:
            logger.info("Cross-Encoder semantic filtering disabled")
            skos_stats.semantic_filtering_disabled = True

        # Update SKOS stats - set concepts_processed here (not in individual vocab methods)
        skos_stats.concepts_processed = sum(len(vocab.get("values", [])) for vocab in vocabularies)

        results = []

        for vocab in vocabularies:
            result = await self._classify_standard_vocabulary(text, vocab, model, temperature, max_tokens, skos_stats)
            if result:
                results.append(result)

        return results

    async def _classify_large_vocabulary_blocks(
        self,
        text: str,
        vocab: dict[str, Any],
        model: str,
        temperature: float,
        max_tokens: int,
        skos_stats: SKOSProcessingStats,
    ) -> ClassificationResult:
        """
        Perform classification using OpenAI API for large vocabularies with block processing.

        This method splits large vocabularies into blocks, finds the best candidates from each block,
        then performs a final refinement round with the top candidates.

        Args:
            text: Text to classify
            vocab: Vocabulary dictionary
            model: OpenAI model to use
            temperature: Temperature setting
            max_tokens: Maximum tokens
            skos_stats: SKOS processing statistics

        Returns:
            ClassificationResult object

        Raises:
            OpenAIError: If OpenAI API call fails
        """
        values = vocab.get("values", [])
        property_name = vocab.get("property", "")

        # Split values into blocks
        block_size = settings.vocab_block_size
        blocks = [values[i : i + block_size] for i in range(0, len(values), block_size)]

        logger.info(
            f"Processing {len(values)} concepts in {len(blocks)} blocks for vocabulary '{property_name}'",
            extra={
                "vocab_name": property_name,
                "total_concepts": len(values),
                "blocks_count": len(blocks),
                "block_size": block_size,
            },
        )

        # Update SKOS stats
        skos_stats.blocks_count = len(blocks)
        skos_stats.block_size = block_size
        skos_stats.block_processing_used = True
        skos_stats.candidates_per_block = settings.max_candidates_per_block

        # Process each block and collect candidates
        all_candidates = []

        for i, block in enumerate(blocks):
            logger.info(
                f"Processing block {i + 1}/{len(blocks)} with {len(block)} concepts for vocabulary '{property_name}'"
            )

            # Create a temporary vocabulary for this block
            block_vocab = {"property": property_name, "values": block}

            # Classify this block
            block_result = await self._classify_single_vocabulary(text, block_vocab, model, temperature, max_tokens)

            # Note: concepts_processed is set elsewhere, don't double-count here

            if block_result and block_result.matches:
                # Keep top candidates from this block
                top_candidates = block_result.matches[: settings.max_candidates_per_block]
                all_candidates.extend(top_candidates)

                logger.info(f"Block {i + 1} produced {len(top_candidates)} candidates")
            else:
                logger.info(f"Block {i + 1} produced no candidates")

        # Always perform final refinement with top candidates from all blocks
        if all_candidates:
            logger.info(
                f"Performing final refinement with {len(all_candidates)} candidates",
                extra={"total_candidates": len(all_candidates), "refinement_limit": settings.final_refinement_limit},
            )

            # Sort by confidence and take top candidates for final refinement
            top_candidates = sorted(all_candidates, key=lambda x: x.confidence, reverse=True)[
                : settings.final_refinement_limit
            ]

            # Create vocabulary with only top candidates for final refinement
            refinement_values = []
            for candidate in top_candidates:
                # Find the original value data
                for value in values:
                    if value["id"] == candidate.id:
                        refinement_values.append(value)
                        break

            refinement_vocab = {
                "name": f"{vocab['name']}_refinement",
                "property": property_name,
                "values": refinement_values,
            }

            # Perform final classification with refined candidates
            final_result = await self._classify_single_vocabulary(
                text, refinement_vocab, model, temperature, max_tokens
            )

            logger.info(
                f"Final refinement completed for vocabulary '{property_name}'",
                extra={"vocab_name": property_name, "final_matches": len(final_result.matches) if final_result else 0},
            )

            # Update SKOS stats
            skos_stats.final_refinement_used = True

            return final_result if final_result else ClassificationResult(property=property_name, matches=[])

        else:
            # No candidates found
            logger.info(f"No candidates found for vocabulary '{property_name}'", extra={"vocab_name": property_name})

            return ClassificationResult(property=property_name, matches=[])

    async def _classify_standard_vocabulary(
        self,
        text: str,
        vocab: dict[str, Any],
        model: str,
        temperature: float,
        max_tokens: int,
        skos_stats: SKOSProcessingStats,
    ) -> ClassificationResult:
        """
        Perform classification using OpenAI API for standard vocabularies.

        Args:
            text: Text to classify
            vocab: Vocabulary dictionary
            model: OpenAI model to use
            temperature: Temperature setting
            max_tokens: Maximum tokens
            skos_stats: SKOS processing statistics

        Returns:
            ClassificationResult object

        Raises:
            OpenAIError: If OpenAI API call fails
        """
        # Note: concepts_processed is already set in _classify_with_openai, don't double-count here

        return await self._classify_single_vocabulary(text, vocab, model, temperature, max_tokens)

    async def _classify_single_vocabulary(
        self, text: str, vocab: dict[str, Any], model: str, temperature: float, max_tokens: int
    ) -> ClassificationResult:
        """
        Perform classification for a single vocabulary using OpenAI API.

        Args:
            text: Text to classify
            vocab: Vocabulary dictionary
            model: OpenAI model to use
            temperature: Temperature setting
            max_tokens: Maximum tokens

        Returns:
            ClassificationResult object

        Raises:
            OpenAIError: If OpenAI API call fails
        """
        # Build prompt for single vocabulary
        prompt = self._build_single_vocabulary_prompt(text, vocab)

        try:
            logger.debug(f"Calling OpenAI API for vocabulary '{vocab['property']}' with model {model}")

            response = await self.openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=temperature,
                max_tokens=max_tokens,
            )

            # Parse response
            content = response.choices[0].message.content
            if content is None:
                raise ValueError("Empty response from OpenAI")
            result_data = json.loads(content)

            # Convert to ClassificationResult object
            matches = []
            for match in result_data.get("matches", []):
                matches.append(
                    ClassificationMatch(
                        id=match.get("id"),
                        label=match.get("value", match.get("label", "")),
                        confidence=match.get("confidence", 0) / 100.0,  # Convert percentage to decimal
                        explanation=match.get("explanation", ""),
                    )
                )

            return ClassificationResult(property=vocab.get("property", ""), matches=matches)

        except Exception as e:
            error_msg = f"OpenAI API call failed for vocabulary '{vocab['property']}': {str(e)}"
            logger.error(error_msg)
            raise OpenAIError(error_msg, {"model": model, "vocabulary": vocab["property"]})

    def _build_single_vocabulary_prompt(self, text: str, vocab: dict[str, Any]) -> str:
        """
        Build the classification prompt for a single vocabulary.

        Args:
            text: Text to classify
            vocab: Vocabulary dictionary

        Returns:
            Formatted prompt string
        """
        property_name = vocab.get("property", "")
        values = vocab.get("values", [])

        prompt = f"""Analyze the following text and determine the most appropriate metadata values from the provided vocabulary.

Text to analyze:
{text}

Vocabulary - Property: {property_name}
Possible values:
"""

        # Add all values for this vocabulary
        for value in values:
            value_line = f"- {value['label']} (ID: {value['id']})\n"

            # Include alternative labels if available
            if "alt_labels" in value and value["alt_labels"]:
                alt_labels_str = ", ".join(value["alt_labels"][:2])
                value_line = f"- {value['label']} (ID: {value['id']}, alt: {alt_labels_str})\n"

            prompt += value_line

        prompt += """
Provide your analysis as a JSON with the following structure:
{
  "matches": [
    {
      "value": "Value Label",
      "id": "Value ID",
      "confidence": 85,
      "explanation": "Explanation for this match"
    }
  ]
}

Include confidence scores (0-100%) for each match and provide a brief explanation for why each value was selected.
Only include matches with confidence scores above 50%.
Focus on the most relevant and specific matches rather than generic ones.
Return the top 5 most relevant matches maximum.
"""

        return prompt

    async def _generate_descriptive_fields(
        self, text: str, model: str, temperature: float, max_tokens: int
    ) -> DescriptiveFields:
        """
        Generate descriptive fields using OpenAI API.

        Args:
            text: Text to generate descriptive fields for
            model: OpenAI model to use
            temperature: Temperature setting
            max_tokens: Maximum tokens

        Returns:
            DescriptiveFields object
        """
        # Build prompt
        prompt = self._build_descriptive_fields_prompt(text)

        try:
            logger.debug(f"Calling OpenAI API with model {model}")

            response = await self.openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=temperature,
                max_tokens=max_tokens,
            )

            # Parse response
            content = response.choices[0].message.content
            if content is None:
                raise ValueError("Empty response from OpenAI")
            result_data = json.loads(content)

            # Convert to DescriptiveFields object with length validation
            title = result_data.get("title", "")[:50]
            short_title = result_data.get("short_title", "")[:25]
            keywords = result_data.get("keywords", [])
            description = result_data.get("description", "")

            # Ensure description meets minimum length requirement
            if len(description) < 200:
                description = description + " " * (200 - len(description))
            # Ensure description doesn't exceed maximum length
            description = description[:500]

            descriptive_fields = DescriptiveFields(
                title=title, short_title=short_title, keywords=keywords, description=description
            )

            return descriptive_fields

        except Exception as e:
            error_msg = f"OpenAI API call failed: {str(e)}"
            logger.error(error_msg)
            raise OpenAIError(error_msg, {"model": model})

    async def _generate_resource_suggestions(
        self, text: str, vocabularies: list[dict[str, Any]], model: str, temperature: float, max_tokens: int
    ) -> ResourceSuggestionFields:
        """
        Generate resource suggestions using OpenAI.

        Args:
            text: Text to analyze for resource suggestions
            vocabularies: List of vocabulary dictionaries
            model: OpenAI model to use
            temperature: Temperature for generation
            max_tokens: Maximum tokens for response

        Returns:
            ResourceSuggestionFields object
        """
        try:
            # Build vocabulary context
            vocab_context = ""
            has_vocabularies = len(vocabularies) > 0

            if has_vocabularies:
                vocab_context = self._build_vocabulary_context(vocabularies)

            # Build prompt
            prompt = self._build_resource_suggestion_prompt(text, vocab_context, has_vocabularies)

            # Call OpenAI
            response = await self.openai_client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "Du bist ein Experte für Bildungsressourcen und hilfst bei der Suche nach passenden Lernmaterialien.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )

            # Parse response
            raw_content = response.choices[0].message.content
            if raw_content is None:
                raise ValueError("Empty response from OpenAI")
            content = raw_content.strip()

            # Extract JSON from response
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                json_content = content[json_start:json_end].strip()
            else:
                json_content = content

            # Parse JSON response
            import json

            result_data = json.loads(json_content)

            # Convert filter suggestions
            filter_suggestions = []
            for filter_data in result_data.get("filter_suggestions", []):
                filter_suggestion = ResourceSuggestionFilterSuggestion(
                    vocabulary_url=filter_data["vocabulary_url"],
                    vocabulary_name=filter_data.get("vocabulary_name"),
                    values=filter_data["values"],
                    uris=filter_data.get("uris", []),
                    labels=filter_data["labels"],
                    confidence=filter_data["confidence"],
                    reasoning=filter_data["reasoning"],
                )
                filter_suggestions.append(filter_suggestion)

            # Validate and truncate lengths if necessary
            title = result_data["title"]
            if len(title) > 50:
                title = title[:47] + "..."

            description = result_data["description"]
            if len(description) > 500:
                description = description[:497] + "..."
            elif len(description) < 200:
                # Pad description if too short
                description = (
                    description
                    + " Diese Ressource unterstützt das Lernen und Verstehen des Themas durch geeignete Materialien und Methoden."
                )
                if len(description) > 500:
                    description = description[:497] + "..."

            # Create resource suggestion fields
            resource_suggestions = ResourceSuggestionFields(
                focus_type=result_data["focus_type"],
                focus_explanation=result_data["focus_explanation"],
                learning_phase=result_data.get("learning_phase"),
                phase_explanation=result_data.get("phase_explanation"),
                filter_suggestions=filter_suggestions,
                search_term=result_data["search_term"],
                keywords=result_data.get("keywords", []),
                title=title,
                description=description,
            )

            return resource_suggestions

        except Exception as e:
            error_msg = f"OpenAI API call failed: {str(e)}"
            logger.error(error_msg)
            raise OpenAIError(error_msg, {"model": model})

    def _build_vocabulary_context(self, vocabularies: list[dict[str, Any]]) -> str:
        """
        Build vocabulary context string for resource suggestion prompts.

        Args:
            vocabularies: List of vocabulary dictionaries

        Returns:
            Formatted vocabulary context string
        """
        if not vocabularies:
            return ""

        context = "Verfügbare Vokabulare:\n\n"

        for vocab in vocabularies:
            vocab_name = vocab.get("name", "Unbekanntes Vokabular")
            vocab_url = vocab.get("url", "")
            values = vocab.get("values", [])

            context += f"**{vocab_name}** ({vocab_url}):\n"

            # Add first 10 values as examples
            for _i, value in enumerate(values[:10]):
                if isinstance(value, dict):
                    label = value.get("label", value.get("id", "Unbekannt"))
                    value_id = value.get("id", "")
                    context += f"- {label} (ID: {value_id})\n"
                else:
                    context += f"- {value}\n"

            if len(values) > 10:
                context += f"... und {len(values) - 10} weitere Werte\n"

            context += "\n"

        return context

    def _build_resource_suggestion_prompt(self, text: str, vocab_context: str, has_vocabularies: bool) -> str:
        """
        Build the resource suggestion prompt for OpenAI.

        Args:
            text: Text to analyze for resource suggestions
            vocab_context: Vocabulary context string
            has_vocabularies: Whether vocabularies are available

        Returns:
            Formatted prompt string
        """
        base_prompt = f"""
Analysiere die folgende Beschreibung einer Lehr- und Lernsituation und generiere passende Suchbegriffe und Filter für eine Bildungsinhalte-Datenbank.

**Beschreibung der Lehr-/Lernsituation:**
{text}

**Fokus-Analyse Regeln:**
1. **INHALTLICH (content-based)**: Wenn ein spezifisches Thema, Fachinhalt oder Sachgebiet im Mittelpunkt steht (z.B. "Australien kennenlernen", "Photosynthese erklären")
2. **KOMPETENZORIENTIERT (competency-based)**: Wenn spezifische Fähigkeiten oder Kompetenzen entwickelt werden sollen (z.B. "Lesekompetenz fördern", "Problemlösefähigkeiten trainieren")
3. **METHODISCH (methodical)**: Wenn pädagogische Methoden, Unterrichtsformen oder didaktische Ansätze im Vordergrund stehen (z.B. "Gruppenarbeit organisieren", "Projektmethode anwenden")

**Lernphasen-Analyse:**
- **introduction**: Einführung in ein neues Thema
- **elaboration**: Vertiefung und Erarbeitung
- **practice**: Übung und Anwendung
- **assessment**: Bewertung und Reflexion
- **null**: Keine spezifische Lernphase erkennbar

**Anweisungen:**
1. Bestimme den Hauptfokus der Beschreibung
2. Identifiziere die Lernphase (falls erkennbar)
3. Generiere einen präzisen Hauptsuchbegriff
4. Erstelle eine Liste relevanter Suchbegriffe/Keywords
5. Formuliere einen aussagekräftigen Titel (max. 50 Zeichen)
6. Schreibe eine detaillierte Beschreibung des Ressourcenbedarfs (200-400 Zeichen)
"""

        if has_vocabularies:
            vocab_prompt = f"""
7. Wähle passende Filter aus den verfügbaren Vokabularen:

{vocab_context}

**Filter-Auswahl Regeln:**
- Bei INHALTLICHEM Fokus: Wähle fachspezifische Disziplinen und inhaltsbezogene Ressourcentypen
- Bei METHODISCHEM Fokus: Wähle methodenbezogene Ressourcentypen und Unterrichtsmaterialien
- Bei KOMPETENZORIENTIERTEM Fokus: Wähle kompetenzfördernde Ressourcentypen

Antworte im folgenden JSON-Format:

```json
{{
  "focus_type": "content-based|competency-based|methodical",
  "focus_explanation": "Begründung für die Fokus-Einschätzung",
  "learning_phase": "introduction|elaboration|practice|assessment|null",
  "phase_explanation": "Begründung für die Lernphasen-Einschätzung",
  "filter_suggestions": [
    {{
      "vocabulary_url": "URL des Vokabulars",
      "vocabulary_name": "Name des Vokabulars",
      "values": ["concept_id_1", "concept_id_2"],
      "labels": ["Label 1", "Label 2"],
      "confidence": 0.85,
      "reasoning": "Begründung für diese Filter-Auswahl"
    }}
  ],
  "search_term": "Hauptsuchbegriff",
  "keywords": ["keyword1", "keyword2", "keyword3"],
  "title": "Kurzer Titel der Ressourcenanfrage (max. 50 Zeichen)",
  "description": "Detaillierte Beschreibung des Ressourcenbedarfs (200-400 Zeichen)"
}}
```
"""
        else:
            vocab_prompt = """

Antworte im folgenden JSON-Format:

```json
{{
  "focus_type": "content-based|competency-based|methodical",
  "focus_explanation": "Begründung für die Fokus-Einschätzung",
  "learning_phase": "introduction|elaboration|practice|assessment|null",
  "phase_explanation": "Begründung für die Lernphasen-Einschätzung",
  "filter_suggestions": [],
  "search_term": "Hauptsuchbegriff",
  "keywords": ["keyword1", "keyword2", "keyword3"],
  "title": "Kurzer Titel der Ressourcenanfrage (max. 50 Zeichen)",
  "description": "Detaillierte Beschreibung des Ressourcenbedarfs (200-400 Zeichen)"
}}
```
"""

        return base_prompt + vocab_prompt

    def _build_descriptive_fields_prompt(self, text: str) -> str:
        """
                Build the descriptive fields prompt for OpenAI.

        {{ ... }}
                    text: Text to generate descriptive fields for

                Returns:
                    Formatted prompt string
        """
        prompt = f"""Analyze the following text and generate descriptive metadata fields.

Text to analyze:
{text}

Generate the following fields based on the content with STRICT character limits:

1. **Title**: A concise, descriptive main title (MAXIMUM 50 characters - count carefully!)
2. **Short Title**: An abbreviated version (MAXIMUM 25 characters - count carefully!)
3. **Keywords**: A list of relevant keywords and key terms from the text
4. **Description**: A comprehensive description (MINIMUM 200, MAXIMUM 500 characters - count carefully!)

IMPORTANT: You MUST strictly adhere to the character limits. Count characters carefully before responding.

Provide your response as a JSON with the following structure:
{{
  "title": "Main title (max 50 chars)",
  "short_title": "Short title (max 25 chars)",
  "keywords": ["keyword1", "keyword2", "keyword3"],
  "description": "Detailed description of the content, highlighting main themes and key information (200-500 chars)"
}}

CRITICAL: Ensure all character limits are strictly followed. The title must be 50 characters or less, short_title must be 25 characters or less, and description must be between 200-500 characters.
"""

        return prompt

    async def close(self) -> None:
        """Close the service and cleanup resources."""
        await self.http_client.aclose()
        await self.openai_client.close()
