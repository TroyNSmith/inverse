Contributing
============

This page describes how to set up a development environment and the tools
used to keep the codebase consistent. Every tool is installed with the ``dev``
dependency group and run through `uv <https://docs.astral.sh/uv/>`_.


Setup
-----

Inverse depends on a local, editable checkout of
`FuelLib <https://github.com/NatLabRockies/FuelLib>`_, which is expected to sit
next to this repository:

.. code-block:: text

   coding/
   ├── FuelLib/
   └── Inverse/

With both repositories cloned, create the virtual environment and install the
git hooks:

.. code-block:: bash

   uv sync
   uv run lefthook install

``uv sync`` installs the package in editable mode together with the ``dev``
dependency group. Inverse requires Python 3.14 or newer.


Tasks
-----

Common commands are defined as `taskipy <https://github.com/taskipy/taskipy>`_
tasks in ``pyproject.toml`` and run with ``uv run task <name>``:

.. table:: Developer tasks
   :widths: auto
   :align: center

   ===============  ==============================  ==========================================
   Task             Command                         Purpose
   ===============  ==============================  ==========================================
   ``format``       ``ruff format``                 Format the code
   ``lint``         ``ruff check --fix``            Lint and fix what can be fixed
   ``types``        ``ty check``                    Type check
   ``imports``      ``lint-imports``                Check the import architecture
   ``test``         ``pytest``                      Run the test suite
   ``pre-commit``   ``lefthook run pre-commit``     Run the full pre-commit suite
   ``build-docs``   ``sphinx-build ...``            Build the HTML documentation
   ``view-docs``    ``open ...``                    Open the built documentation
   ===============  ==============================  ==========================================

Run ``uv run task --list`` to see all tasks.


Pre-commit hooks
----------------

`Lefthook <https://lefthook.dev>`_ runs the following commands, in order, before
every commit (see ``lefthook.yaml``):

#. ``ruff format``
#. ``ruff check --fix``
#. ``ty check``
#. ``lint-imports``
#. ``pytest``
#. ``git diff --exit-code``

The last step fails the commit if any earlier step modified files, e.g. if the
formatter rewrote a staged file. Review and stage those changes, then commit
again. Run the same suite without committing with ``uv run task pre-commit``.


Formatting and linting
----------------------

`Ruff <https://docs.astral.sh/ruff/>`_ both formats and lints the code. It is
configured in ``ruff.toml``. The formatter follows Black's style with an
88-character line length. The linter enforces:

.. table:: Enabled lint rules
   :widths: auto
   :align: center

   ======================  ==============================================================
   Rules                   Enforces
   ======================  ==============================================================
   ``D``                   Docstrings are present and formatted (Google convention)
   ``DOC``                 Docstrings match the signature (``Args``, ``Returns``, ...)
   ``ANN``                 Type annotations on all functions
   ``E501``                Line length
   ``INP001``              ``__init__.py`` in every package
   ``I001``                Sorted imports
   ``RUF022``              Sorted ``__all__``
   ``F401``, ``F841``      No unused imports or variables
   ``F403``, ``F405``      No wildcard imports
   ======================  ==============================================================

Docstring rules apply to tests and fixtures as well, so a fixture needs a
``Returns:`` section. Private helpers (``_name``) are exempt. The ``examples``
directory and ``docs`` configuration are not linted.


Type checking
-------------

`ty <https://docs.astral.sh/ty/>`_ checks ``src`` and ``tests`` against the
project virtual environment (``.venv``), with ``../FuelLib`` added to the search
path. It is configured under ``[tool.ty]`` in ``pyproject.toml``. Array shapes
and dtypes can be annotated with `jaxtyping <https://docs.kidger.site/jaxtyping/>`_.


Import architecture
-------------------

`Import Linter <https://import-linter.readthedocs.io>`_ enforces a layered
architecture within the ``inverse`` package (see ``[tool.importlinter]`` in
``pyproject.toml``). A module may import from layers below it, but never from
layers above it. Modules in the same layer, separated by ``|``, must not
import each other.

.. table:: Import layers, from highest to lowest
   :widths: auto
   :align: center

   =======  ===========================================
   Layer    Modules
   =======  ===========================================
   1        ``output`` | ``optimize``
   2        ``problem``
   3        ``mix`` | ``models`` | ``plot`` | ``utils``
   =======  ===========================================

For example, ``problem`` may import ``mix`` and ``models``, but ``mix`` may not
import ``problem``. When adding a module, add it to the appropriate layer.


Testing
-------

Tests live in ``tests`` and run with `pytest <https://docs.pytest.org>`_. Shared
fixtures in ``tests/conftest.py`` provide a two-component fuel
(``heptane-decane``, no aromatics) and a full jet fuel (``posf10264``, 67
components with aromatics), along with a :class:`~inverse.problem.Problem` for
each. ``test_accuracy.py`` checks the mixing rules in :mod:`inverse.mix`
against FuelLib's own correlations.

Measure coverage with `pytest-cov <https://pytest-cov.readthedocs.io>`_:

.. code-block:: bash

   uv run pytest --cov=inverse --cov-report=term-missing


Documentation
-------------

The documentation is built with `Sphinx <https://www.sphinx-doc.org>`_ and the
Read the Docs theme:

.. code-block:: bash

   uv run task build-docs
   uv run task view-docs

API pages are generated from docstrings with ``autodoc`` and ``napoleon``, so
Google-style docstrings render directly. References are kept in
``docs/refs.bib`` and cited with ``:footcite:p:`key``` via
`sphinxcontrib-bibtex <https://sphinxcontrib-bibtex.readthedocs.io>`_. Each
page that cites a reference ends with a ``.. footbibliography::`` directive.


Changelog and releases
----------------------

Record every notable change under ``[Unreleased]`` in ``CHANGELOG.md``,
following `Keep a Changelog <https://keepachangelog.com/en/1.1.0/>`_. Versions
follow `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.
