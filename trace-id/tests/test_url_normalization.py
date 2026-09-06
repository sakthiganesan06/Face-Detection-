"""
Tests for URL normalization utilities.

Verifies:
  - Scheme normalization
  - Trailing slash removal
  - Tracking parameter stripping
  - Domain extraction
  - Social media classification
"""

from __future__ import annotations

import pytest

from app.search.base import normalize_url, extract_domain, classify_candidate_type


class TestNormalizeUrl:
    def test_strips_fragment(self):
        url = "https://example.com/page#section"
        assert "#section" not in normalize_url(url)

    def test_lowercase_scheme_and_host(self):
        url = "HTTPS://Example.COM/path"
        result = normalize_url(url)
        assert result.startswith("https://example.com")

    def test_removes_trailing_slash(self):
        url = "https://example.com/path/"
        result = normalize_url(url)
        assert not result.endswith("/") or result == "https://example.com/"

    def test_strips_utm_params(self):
        url = "https://example.com/page?utm_source=google&utm_medium=cpc&content=real"
        result = normalize_url(url)
        assert "utm_source" not in result
        assert "utm_medium" not in result
        # content= is not a tracking param
        assert "content=real" in result

    def test_strips_fbclid(self):
        url = "https://example.com/page?fbclid=abc123"
        result = normalize_url(url)
        assert "fbclid" not in result

    def test_adds_https_scheme(self):
        url = "example.com/path"
        result = normalize_url(url)
        assert result.startswith("https://")

    def test_empty_url_returns_empty(self):
        assert normalize_url("") == ""

    def test_preserves_query_params(self):
        url = "https://example.com/search?q=test&page=2"
        result = normalize_url(url)
        assert "q=test" in result
        assert "page=2" in result

    def test_idempotent(self):
        url = "https://example.com/path"
        assert normalize_url(normalize_url(url)) == normalize_url(url)


class TestExtractDomain:
    def test_simple_domain(self):
        assert extract_domain("https://example.com/page") == "example.com"

    def test_strips_www(self):
        assert extract_domain("https://www.example.com/page") == "example.com"

    def test_subdomain_preserved(self):
        # Only strips 'www.' prefix
        domain = extract_domain("https://images.example.com/page")
        assert domain == "images.example.com"

    def test_x_com(self):
        assert extract_domain("https://x.com/user/status/123") == "x.com"

    def test_instagram(self):
        assert extract_domain("https://www.instagram.com/p/abc/") == "instagram.com"

    def test_empty_url_returns_empty(self):
        assert extract_domain("") == ""

    def test_strips_port(self):
        domain = extract_domain("https://example.com:8080/path")
        assert "8080" not in domain
        assert domain == "example.com"

    def test_no_scheme(self):
        domain = extract_domain("x.com/user")
        assert domain == "x.com"


class TestClassifyCandidateType:
    def test_x_com_is_social(self):
        assert classify_candidate_type("x.com") == "social"

    def test_twitter_is_social(self):
        assert classify_candidate_type("twitter.com") == "social"

    def test_instagram_is_social(self):
        assert classify_candidate_type("instagram.com") == "social"

    def test_facebook_is_social(self):
        assert classify_candidate_type("facebook.com") == "social"

    def test_linkedin_is_social(self):
        assert classify_candidate_type("linkedin.com") == "social"

    def test_threads_is_social(self):
        assert classify_candidate_type("threads.net") == "social"

    def test_example_is_web(self):
        assert classify_candidate_type("example.com") == "web"

    def test_news_site_is_web(self):
        assert classify_candidate_type("bbc.co.uk") == "web"

    def test_empty_domain_is_web(self):
        assert classify_candidate_type("") == "web"

    def test_subdomain_of_social(self):
        # Subdomain of a social platform
        assert classify_candidate_type("mobile.twitter.com") == "social"
