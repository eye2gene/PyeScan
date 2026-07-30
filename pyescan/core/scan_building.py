import logging

from pyescan.core.image import LazyImage
from pyescan.core.scan_enface import FAFScan, IRScan
from pyescan.core.scan_oct import BScan, BScanArray, OCTScan

logger = logging.getLogger(__name__)

KNOWN_MODALITIES = {
    'OCT', 'SLO - Infrared', 'AF - Blue', 'Color Fundus',
}


class ScanBuildError(Exception):
    """Raised when a scan cannot be constructed from metadata."""
    pass


def _get_metadata_attr(meta, attr_name, context=""):
    """
    Safely get an attribute from a metadata view with clear error on failure.
    """
    try:
        return getattr(meta, attr_name)
    except AttributeError as e:
        raise ScanBuildError(
            f"Cannot build scan: required metadata attribute '{attr_name}' is missing. "
            f"{f'Context: {context}. ' if context else ''}"
            f"Original error: {e}"
        ) from e


def build_oct_from_metadata(oct_meta, enface_meta=None):
    source_id = _get_metadata_attr(oct_meta, 'source_id', context="OCT scan")

    if enface_meta:
        enface_img_path = _get_metadata_attr(
            enface_meta, 'image_location',
            context=f"enface for OCT source_id={source_id}"
        )
        enface_img = LazyImage(enface_img_path)
        enface = IRScan(image=enface_img, metadata=enface_meta)
    else:
        enface = None

    # Get bscan metadata list
    try:
        bscan_metas = oct_meta.bscans
    except AttributeError as e:
        raise ScanBuildError(
            f"Cannot build OCT scan (source_id={source_id}): "
            f"failed to retrieve B-scan list from metadata. "
            f"This requires 'n_images' to be available. Original error: {e}"
        ) from e

    if not bscan_metas:
        raise ScanBuildError(
            f"OCT scan (source_id={source_id}) has no B-scans. "
            f"The metadata reports 0 images for this scan."
        )

    bscans = list()
    for i, bscan_meta in enumerate(bscan_metas):
        bscan_img_path = _get_metadata_attr(
            bscan_meta, 'image_location',
            context=f"B-scan index {i} of OCT source_id={source_id}"
        )
        bscan_img = LazyImage(bscan_img_path)
        bscan = BScan(bscan_img, i, bscan_meta)
        bscans.append(bscan)

    bscan_array = BScanArray(bscans)
    scan = OCTScan(enface, bscan_array, metadata=oct_meta)
    return scan


def build_faf_from_metadata(scan_meta):
    source_id = _get_metadata_attr(scan_meta, 'source_id', context="FAF scan")
    scan_img_path = _get_metadata_attr(
        scan_meta, 'image_location',
        context=f"FAF scan source_id={source_id}"
    )
    scan_img = LazyImage(scan_img_path)
    scan = FAFScan(image=scan_img, metadata=scan_meta)
    return scan


def build_ir_from_metadata(scan_meta):
    source_id = _get_metadata_attr(scan_meta, 'source_id', context="IR scan")
    scan_img_path = _get_metadata_attr(
        scan_meta, 'image_location',
        context=f"IR scan source_id={source_id}"
    )
    scan_img = LazyImage(scan_img_path)
    scan = IRScan(image=scan_img, metadata=scan_meta)
    return scan


def build_from_metadata(metadata):
    """
    Build scan objects from a top-level MetadataView.

    Requires the metadata to provide:
    - n_scans: number of scan entries
    - For each scan: group, modality, source_id
    - For OCT scans: n_images, image_location for each B-scan
    - For enface/FAF/IR scans: image_location

    Parameters
    ----------
    metadata : MetadataView
        Top-level metadata view with a configured parser.

    Returns
    -------
    list[BaseScan]
        Constructed scan objects.

    Raises
    ------
    ScanBuildError
        If required metadata attributes are missing or inaccessible.
    ValueError
        If no scans could be built from any group.
    """
    # Validate we can get the group structure
    try:
        groups = metadata.get_groups()
    except AttributeError as e:
        raise ScanBuildError(
            f"Cannot build scans: failed to retrieve scan groups from metadata. "
            f"This requires 'n_scans' and 'group' to be available. "
            f"Original error: {e}"
        ) from e

    if not groups:
        raise ScanBuildError(
            "No scan groups found in the metadata. "
            "The metadata appears to contain 0 scans."
        )

    scans = list()
    skipped_groups = []
    build_errors = []

    for group in groups:
        try:
            modalities = [_get_metadata_attr(scan, 'modality', context="scan group") for scan in group]
        except ScanBuildError as e:
            build_errors.append(str(e))
            continue

        try:
            if modalities == ['SLO - Infrared', 'OCT']:
                ir_meta, oct_meta = group
                scan = build_oct_from_metadata(oct_meta, ir_meta)
                scans.append(scan)
                
            elif modalities == ['OCT', 'SLO - Infrared']:
                oct_meta, ir_meta = group
                scan = build_oct_from_metadata(oct_meta, ir_meta)
                scans.append(scan)
                
            elif modalities == ['OCT']:
                oct_meta = group[0]
                scan = build_oct_from_metadata(oct_meta, None)
                scans.append(scan)
                
            elif modalities == ['AF - Blue']:
                scan_meta = group[0]
                scan = build_faf_from_metadata(scan_meta)
                scans.append(scan)
                
            elif modalities == ['SLO - Infrared']:
                scan_meta = group[0]
                scan = build_ir_from_metadata(scan_meta)
                scans.append(scan)
                
            else:
                skipped_groups.append(modalities)
                logger.warning(f"Skipping unrecognised scan group with modalities: {modalities}")

        except ScanBuildError as e:
            build_errors.append(str(e))
            logger.error(f"Failed to build scan group {modalities}: {e}")

    if not scans:
        error_details = []
        if skipped_groups:
            error_details.append(
                f"Unrecognised modality combinations: {skipped_groups}. "
                f"Supported: {sorted(KNOWN_MODALITIES)}"
            )
        if build_errors:
            error_details.append(
                f"Build errors ({len(build_errors)}):\n" +
                "\n".join(f"    • {e}" for e in build_errors)
            )
        raise ValueError(
            "No scans could be built from the metadata.\n" +
            "\n".join(error_details)
        )

    if build_errors:
        logger.warning(
            f"{len(build_errors)} scan group(s) failed to build. "
            f"Successfully built {len(scans)} scan(s)."
        )

    return scans