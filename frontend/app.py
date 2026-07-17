"""LLMOps Studio Frontend Entrypoint.

Orchestrates application setup, validates environmental configurations,
constructs the sidebar layout, and delegates page views rendering.
"""

import sys
from pathlib import Path
import streamlit as st

# Ensure backend directory is in the python path
workspace_root = Path(__file__).resolve().parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from backend.core.config import get_settings, Settings
from backend.utils.fs import run_startup_validation
from backend.utils.logging import initialize_logging


def initialize_application() -> None:
    """Initializes the backend logging systems and directory storage structures."""
    # 1. Set up multi-file rotating logging streams
    initialize_logging()

    # 2. Run system directories existence and writability validations
    try:
        run_startup_validation()
    except Exception as e:
        st.error(f"Application initialization failure: {e}")
        st.stop()


def validate_environment(settings: Settings) -> None:
    """Ensures configuration settings loaded successfully.

    Args:
        settings: Active settings config class.
    """
    if not settings.workspace_root.exists():
        st.error(
            f"Startup validation error: Workspace root does not exist at '{settings.workspace_root}'"
        )
        st.stop()


def build_navigation() -> st.navigation:
    """Configures multi-page mappings and returns the Streamlit navigation routing handler."""
    dashboard_page = st.Page(
        "pages/dashboard.py", 
        title="Dashboard", 
        icon="📊", 
        default=True
    )
    datasets_page = st.Page(
        "pages/datasets.py", 
        title="Datasets", 
        icon="📁"
    )
    training_page = st.Page(
        "pages/training.py", 
        title="Training", 
        icon="⚙️"
    )
    evaluation_page = st.Page(
        "pages/evaluation.py", 
        title="Evaluation", 
        icon="🧪"
    )
    benchmarks_page = st.Page(
        "pages/benchmarks.py", 
        title="Benchmarks", 
        icon="📈"
    )
    experiments_page = st.Page(
        "pages/experiments.py", 
        title="Experiments", 
        icon="🔬"
    )
    registry_page = st.Page(
        "pages/registry.py", 
        title="Model Registry", 
        icon="🗄️"
    )
    settings_page = st.Page(
        "pages/settings.py", 
        title="Settings", 
        icon="🔧"
    )

    return st.navigation([
        dashboard_page,
        datasets_page,
        training_page,
        evaluation_page,
        benchmarks_page,
        experiments_page,
        registry_page,
        settings_page
    ])


def render_sidebar(settings: Settings) -> None:
    """Draws sidebar header headers, confirmation badges, and environment states.

    Args:
        settings: Active settings config class.
    """
    st.sidebar.markdown(f"### {settings.app_name}")
    st.sidebar.success("✓ System Startup Validated")
    st.sidebar.caption(f"Environment: **{settings.app_env}**")
    st.sidebar.caption(f"App Version: **{settings.app_version}**")
    st.sidebar.markdown("---")


def main() -> None:
    """Main application loop."""
    # 1. Initialize loggers and directories
    initialize_application()

    # 2. Fetch config and validate path variables
    settings = get_settings()
    validate_environment(settings)

    # 3. Render sidebar configuration widgets
    render_sidebar(settings)

    # 4. Bind and execute navigation page loading
    pg = build_navigation()
    pg.run()


if __name__ == "__main__":
    main()
