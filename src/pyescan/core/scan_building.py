from pyescan.core.image import LazyImage

from pyescan.core.scan_enface import IRScan, FAFScan
from pyescan.core.scan_oct import BScan, BScanArray, OCTScan
    

def build_oct_from_metadata(oct_meta, enface_meta=None):
    
    if enface_meta:
        enface_img_path = enface_meta.image_location
        enface_img = LazyImage(enface_img_path)
        enface = IRScan(image=enface_img, metadata=enface_meta)
    else:
        enface = None

    bscans = list()
    for i, bscan_meta in enumerate(oct_meta.bscans):
        bscan_img_path = bscan_meta.image_location
        bscan_img = LazyImage(bscan_img_path)
        bscan = BScan(bscan_img, i, bscan_meta)
        bscans.append(bscan)
    bscan_array = BScanArray(bscans)
    
    scan = OCTScan(enface, bscan_array, metadata=oct_meta)
    return scan

def build_faf_from_metadata(scan_meta):
    scan_img_path = scan_meta.image_location
    scan_img = LazyImage(scan_img_path)
    scan = FAFScan(image=scan_img, metadata=scan_meta)
    return scan

def build_ir_from_metadata(scan_meta):
    scan_img_path = scan_meta.image_location
    scan_img = LazyImage(scan_img_path)
    scan = IRScan(image=scan_img, metadata=scan_meta)
    return scan

def build_from_metadata(metadata):
    scans = list()
    for group in metadata.get_groups():
        modalities = [scan.modality for scan in group]

        if modalities == ['SLO - Infrared', 'OCT']:
            ir_meta, oct_meta = group
            scan = build_oct_from_metadata(oct_meta, ir_meta)
            scans.append(scan)
            
        elif modalities == ['OCT', 'SLO - Infrared' ]:
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
            print("Skipping Group", modalities)
    return scans
    
    