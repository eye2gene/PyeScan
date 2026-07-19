import colorsys
import io
from ipywidgets import widgets
import numpy as np
from PIL import Image as PILImage, ImageDraw, ImageFont

def _encode_image(image, default="blank"):
    # Save image to buffer
    imgByteArr = io.BytesIO()
    if image is None:
        image = PILImage.new('L', (20, 20))
    image.save(imgByteArr, format='PNG')
    
    # Turn the BytesIO object back into a bytes object
    imgByteArr = imgByteArr.getvalue()
    return imgByteArr


def image_array_display_widget(images, width=320, height=320, return_slider=False):
    encoded_volume = [ _encode_image(image) for image in images ]
    n_images = len(encoded_volume)

    # Create a slider widget for image navigation
    w_slider = widgets.IntSlider(min=0, max=n_images-1, step=1,
                                 layout={'width': str(width)+'px'},
                                 readout=True,
                                 readout_format='d')

    w_value = widgets.Label(value="Some text",
                            layout={'width': str(width)+'px', 'visible': 'true'})

    # Create an image widget for displaying images
    w_image_volume = widgets.Image(value=encoded_volume[n_images//2], width=width, height=height)

    # Define a function to update the displayed image based on the slider value
    def update_image(change):
        index = change.new
        w_image_volume.value=encoded_volume[index]

    # Connect the slider and image widgets
    w_slider.observe(update_image, names='value')

    # Arrange the widgets using a VBox
    display_layout = widgets.VBox([w_slider, w_image_volume])

    if return_slider:
        return display_layout, w_slider
    else:
        return display_layout

def draw_bscan_lines(enface_image, bscan_positions, bscan_index=None, draw_pos=False):
    enface_image = enface_image.copy().convert("RGB")
    width, height = enface_image.size
    
    draw = ImageDraw.Draw(enface_image)
    for i, bscan_position in enumerate(bscan_positions):
        color = "white" if i == bscan_index else "lime"
        width = 7 if i == bscan_index else 3
        
        # Fix indexing from bottom left to top left
        #pos =  ( (bscan_position[0][0], height - bscan_position[0][1]),
        #         (bscan_position[1][0], height - bscan_position[1][1]) )
        
        pos =  ( (bscan_position[0][0], bscan_position[0][1]),
                 (bscan_position[1][0], bscan_position[1][1]) )
        draw.line(pos, fill=color, width=width)
    
    if draw_pos:
        draw.text((0,0), f'{bscan_index:d}')
    
    return enface_image

def enface_display_widget(image, width=320, height=320):
    encoded_enface = _encode_image(image)
    w_image_enface = widgets.Image(value=encoded_enface, width=width, height=height)
    return w_image_enface

def oct_display_widget(images, enface_image, bscan_locations=None, width=640, height=320, enface_size=320):

    # Create an image widget for displaying images
    encoded_enface = _encode_image(enface_image)
    w_image_enface = widgets.Image(value=encoded_enface, width=enface_size, height=enface_size)
    
    # Get widget for displaying volume
    w_image_volume, w_slider = image_array_display_widget(images, width=width, height=height, return_slider=True)
    
    if not (bscan_locations is None):
        def update_enface(change):
            index = change.new
            updated_enface = draw_bscan_lines(enface_image, bscan_locations, index)
            w_image_enface.value = _encode_image(updated_enface)
        w_slider.observe(update_enface, names='value')
        update_enface(type('change', (), {'new': 0})())
    
    # Create layout
    display_layout = widgets.HBox([w_image_enface, w_image_volume])

    return display_layout
    
def generate_distinct_colors(n):
    """
    Generate n distinct colors.
    """
    colors = []
    for i in range(n):
        hue = i / n
        saturation = 0.7 + (i % 3) * 0.1  # Slight variation in saturation
        value = 0.8 + (i % 2) * 0.2  # Slight variation in value
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append(tuple(int(x * 255) for x in rgb))
    return colors

def overlay_rgba_images(image_list):
    """
    Overlay multiple RGBA images with equal weighting
    Each image should be (h, w, 4) numpy array
    """
    # Separate alpha from RGB
    rgbs = [img[..., :3] for img in image_list]
    alphas = [img[..., 3:] / 255.0 for img in image_list]  # normalize alpha to 0-1
    
    # Calculate combined alpha (taking care to not divide by zero)
    total_alpha = np.sum(alphas, axis=0)
    total_alpha = np.clip(total_alpha, 1e-8, None)  # avoid division by zero
    
    # Weight each RGB by its alpha and sum
    weighted_rgb = np.sum([(rgb * alpha) for rgb, alpha in zip(rgbs, alphas)], axis=0)
    
    # Normalize by total alpha
    final_rgb = weighted_rgb / total_alpha
    
    # Combine with new alpha channel (maximum of alphas)
    final_alpha = np.max(alphas, axis=0) * 255
    final_rgba = np.concatenate([final_rgb, final_alpha], axis=-1)
    
    return final_rgba.astype(np.uint8)

def overlay_masks(image, masks, colors=None, feature_names=None, alpha=0.5, max_height=None):
    """
    Overlay multiple binary masks on a grayscale image with different transparent colors.
    
    :param image: PIL Image in grayscale mode
    :param masks: List of binary masks (numpy arrays or PIL Images)
    :param colors: Optional list of RGB tuples. If not provided, colors will be auto-generated.
    :param alpha: Transparency of the overlay (0-1), default is 0.5
    :return: PIL Image with overlaid masks
    """
    # TODO: Change to use overlay_rgba_images to unify with oct overlay
    # OCTScan (or other) should take owenership of the making of the coloured masks and then
    #   overlay them using the helper function.
    
    # Convert grayscale to RGB
    rgb_image = image.convert('RGB')
    
    # Get image dimensions
    width, height = rgb_image.size

    if max_height and height > max_height:
        width,height  = int(max_height * width / height), max_height
        rgb_image = rgb_image.resize((width, height), PILImage.NEAREST)
        
    
    # Convert image to numpy array
    img_array = np.array(rgb_image)

    # Generate colors if not provided
    if colors is None:
        colors = generate_distinct_colors(len(masks))

    # Create an empty array for the overlay
    overlay = np.zeros_like(img_array, dtype=np.float32)

    # Process each mask and color
    for mask, color in zip(masks, colors):
        
        # Skip missing masks
        if mask is None: continue
            
        # Ensure mask is a PIL Image
        if isinstance(mask, np.ndarray):
            mask = PILImage.fromarray(mask)
            
        mask = mask.convert('L')
            
        # Resize mask if it doesn't match the input image size
        if mask.size != (width, height):
            mask = mask.resize((width, height), PILImage.NEAREST)
            
        # Convert mask to numpy array and normalize to range [0, 1]
        mask_array = np.array(mask).astype(np.float32) / 255.0
        
        # Add colored mask to overlay
        for c in range(3):  # RGB channels
            overlay[:,:,c] += mask_array * color[c] / 255.0

    # Apply alpha blending
    overlay = np.clip(overlay * alpha, 0, 1)

    # Blend original image with overlay
    result = img_array * (1 - overlay) + (overlay * 255)
    result = result.astype(np.uint8)
    
    # Convert back to PIL Image for drawing text
    result_image = PILImage.fromarray(result)
    draw = ImageDraw.Draw(result_image)
    
    if feature_names: 
        # Use default font
        font = ImageFont.load_default()
        
        # Draw labels
        y_offset = 10
        for name, color in zip(feature_names, colors):
            # Draw color swatch
            draw.rectangle([10, y_offset, 30, y_offset + 20], fill=color, outline='white')
            # Draw text
            draw.text((35, y_offset), name, font=font, fill='white', stroke_width=2, stroke_fill='black')
            y_offset += 30

    return result_image

def render_volume_data(data, color=(255, 0, 0), heatmap=True, contours=True, figsize=(10,10)):
    """
    Create either a heatmap or contour overlay image
    Uses matplotlib to render
    Returns: RGBA numpy array (h x w x 4)
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    
    cmap = mcolors.LinearSegmentedColormap.from_list('custom', [(0,0,0), np.array(color)/255])
        
    # Create figure without displaying it
    fig, ax = plt.subplots(1, figsize=figsize)

    n, h, w = data.shape
    X, Y = np.meshgrid(np.arange(w), np.arange(n))
    
    mask = data.any(axis=1)
    thickness = data.mean(axis=1)

    if heatmap:
        im = ax.imshow(thickness, aspect='auto', cmap=cmap, interpolation='nearest', alpha=mask*1.0)
    else:
        im = ax.imshow(mask, aspect='auto', cmap=cmap, interpolation='nearest', alpha=mask*1.0)
        
    #im_type = im_type.lower()
    #if im_type == "contours": # Causes havok with flipping
        #masked_thickness = np.ma.masked_where(thickness==0, thickness)
        #im = ax.contourf(X, Y, masked_thickness[::-1,...], levels=5, cmap=cmap)
        
    if contours:
        ax.contour(X, Y, thickness, levels=5, colors='black', linewidths=5) 
    
    ax.set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    #plt.margins(0,0)
    
    # Convert to image
    fig.canvas.draw()

    # Get the RGBA buffer
    data = np.asarray(fig.canvas.buffer_rgba())
    
    plt.close()  # Clean up
    return data