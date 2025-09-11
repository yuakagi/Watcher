# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "watcher"
copyright = "2025, Yu Akagi, MD"
author = "Yu Akagi, MD"
release = "0.0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_sitemap",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_extra_path = ["_static/robots.txt"]  # Add robots.txt to the root of the built docs

# -- Extended by the author --
html_show_sourcelink = False
autodoc_default_options = {}
html_css_files = [
    "color_theme.css",
]

html_logo = "_static/images/logos/logo.svg"
html_favicon = "_static/images/logos/favicon.svg"
html_theme_options = {
    # Show logo only on the sidebar top
    "logo_only": True,
    # Hide the version string for Sphinx
    "display_version": False,
    # Collapse the navigation sidebar (`True` is usually inconvenient)
    "collapse_navigation": False,
    # Sticky navigation
    "sticky_navigation": True,
    # Navigation depth in the sidebar. Set sufficiently deep!
    "navigation_depth": 10,
    # If True, only page titles are shown in the sidebar (usually this is fine; set to False if you want to show section headings as well)
    "titles_only": True,
}


# Sitemap
# ⚠️ Make sure to change this if you change the domain or project name!
# ⚠️ Also, change `_static/robots.txt` accordingly! Do not forget robots.txt.
# ⚠️ If you want to make your site appear in Google search results, set up Google Search Console (https://search.google.com/search-console/welcome).
html_baseurl = "https://yuakagi.github.io/Watcher/"  # Tailing slash is important!!!
sitemap_filename = "sitemap.xml"
sitemap_url_scheme = "{link}"  # ensures clean URLs
