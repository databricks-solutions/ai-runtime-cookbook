"""Unity Catalog checks for the Ray head's model registration target."""
from importlib.metadata import version

from databricks.sdk.errors import AlreadyExists, DatabricksError, PermissionDenied, ResourceAlreadyExists
from databricks.sdk.service.catalog import CatalogType


# The SDK raises TimeoutError when its HTTP retry budget expires.
METADATA_ERRORS = (DatabricksError, TimeoutError, ConnectionError)


def _read(get, name, missing_code=None):
    try:
        return get(name, include_browse=True), False
    except METADATA_ERRORS as exc:
        if missing_code and getattr(exc, "error_code", None) == missing_code:
            return None, True
        print(f"Could not inspect {name}: {exc}", flush=True)
        return None, False


def _effective_grants(workspace, principal, kind, name):
    """Only positive, caller-filtered evidence is useful; absent rows can hide group ownership."""
    grants = set()
    if principal:
        page_token = None
        try:
            while True:
                page = workspace.grants.get_effective(
                    kind, name, principal=principal, max_results=0, page_token=page_token,
                )
                for assignment in page.privilege_assignments or []:
                    for item in assignment.privileges or []:
                        if item.privilege:
                            grants.add(item.privilege.value)
                page_token = page.next_page_token
                if not page_token:
                    break
        except METADATA_ERRORS as exc:
            print(f"Could not inspect effective privileges on {name}: {exc}", flush=True)
    return grants


def preflight_registration(workspace, catalog, schema, model):
    """Fail on a known unusable target and report permissions that metadata cannot confirm."""
    full_schema = f"{catalog}.{schema}"
    configure = f"Set uc_catalog/uc_schema and registered_model_name to a writable target for {model}."
    print(f"Checking Unity Catalog registration target {model} before tuning "
          f"(databricks-sdk {version('databricks-sdk')}).", flush=True)
    catalog_info, catalog_missing = _read(workspace.catalogs.get, catalog, "CATALOG_DOES_NOT_EXIST")
    if catalog_missing:
        raise RuntimeError(f"Catalog {catalog} was not found for this run. Set uc_catalog to an existing writable catalog.")
    if catalog_info and catalog_info.catalog_type == CatalogType.DELTASHARING_CATALOG:
        raise RuntimeError(f"Catalog {catalog} is read-only (Delta Sharing). {configure}")
    if catalog_info and catalog_info.catalog_type in {CatalogType.FOREIGN_CATALOG, CatalogType.SYSTEM_CATALOG}:
        raise RuntimeError(
            f"Catalog {catalog} ({catalog_info.catalog_type.value}) does not support model registration. "
            f"{configure}"
        )
    if catalog_info and catalog_info.browse_only is True:
        raise RuntimeError(f"Catalog {catalog} is browse-only for this caller; USE CATALOG is required. {configure}")
    schema_info, schema_missing = _read(workspace.schemas.get, full_schema, "SCHEMA_DOES_NOT_EXIST")
    if schema_info and schema_info.browse_only is True:
        raise RuntimeError(
            f"Schema {full_schema} is browse-only for this caller; registration requires both "
            f"USE CATALOG on {catalog} and USE SCHEMA on {full_schema}. {configure}"
        )

    try:
        identity = workspace.current_user.me()
        caller = identity.user_name or getattr(identity, "application_id", None)
        owners = {caller} if caller else set()
        if caller:
            owners.update(group.display for group in identity.groups or [] if group.display)
    except METADATA_ERRORS as exc:
        print(f"Could not identify the registration caller: {exc}", flush=True)
        caller, owners = None, set()

    unconfirmed = []

    def check(info, kind, name, requirements):
        if info and info.owner in owners:
            return
        grants = _effective_grants(workspace, caller, kind, name)
        if "ALL_PRIVILEGES" in grants:
            return
        for alternatives in requirements:
            if grants.isdisjoint(alternatives):
                label = " or ".join(value.replace("_", " ") for value in alternatives)
                condition = " if the schema needs creation" if alternatives == ("CREATE_SCHEMA",) else ""
                unconfirmed.append(f"{label} on {name}{condition}")

    catalog_requirements = [("USE_CATALOG",)]
    if schema_info is None:
        catalog_requirements.append(("CREATE_SCHEMA",))
    check(catalog_info, "catalog", catalog, catalog_requirements)
    if schema_missing:
        print(f"Schema {full_schema} was not found; it will be created if needed after tuning.", flush=True)
    else:
        # MLflow tries to create the model before adding a version, even when the model exists.
        check(schema_info, "schema", full_schema, [("USE_SCHEMA",), ("CREATE_MODEL", "CREATE_FUNCTION")])
        model_info, model_missing = _read(workspace.registered_models.get, model, "RESOURCE_DOES_NOT_EXIST")
        if model_info:
            if model_info.owner not in owners:
                grants = _effective_grants(workspace, caller, "function", model)
                if grants.isdisjoint({"ALL_PRIVILEGES", "CREATE_MODEL_VERSION"}):
                    unconfirmed.append(f"ownership or CREATE MODEL VERSION on {model} to add a version")
        else:
            state = "not found" if model_missing else "visibility unknown"
            print(f"Registered model {model}: {state}. If it exists, confirm ownership or CREATE MODEL "
                  "VERSION to add a version.", flush=True)
    if unconfirmed:
        print("Before registration, confirm " + "; ".join(unconfirmed) +
              ". Read-only metadata could not establish these privileges; continuing.", flush=True)


def ensure_uc_schema(workspace, catalog, schema):
    full_name = f"{catalog}.{schema}"
    info, _ = _read(workspace.schemas.get, full_name, "SCHEMA_DOES_NOT_EXIST")
    if info is not None:
        return
    try:
        workspace.schemas.create(name=schema, catalog_name=catalog)
        print(f"Created schema {full_name}")
    except PermissionDenied as exc:
        exc.add_note(
            f"Cannot create schema {full_name}. Check USE CATALOG and CREATE SCHEMA on a writable "
            "catalog, or set uc_catalog/uc_schema to an existing writable schema."
        )
        raise
    except (AlreadyExists, ResourceAlreadyExists):
        pass
    except DatabricksError as exc:
        if exc.error_code != "SCHEMA_ALREADY_EXISTS":
            raise
