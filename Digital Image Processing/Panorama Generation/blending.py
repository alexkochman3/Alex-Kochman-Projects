def blend_images(img1, img2, translation_dist):
    """
    Blend the stitched image to smooth the seams.
    Args:
        img1 (ndarray): First image.
        img2 (ndarray): Second image to blend into img1.
        translation_dist (tuple): Translation distances (x, y) applied to img1 during stitching.
    Returns:
        blended_img (ndarray): The blended panorama image.
    """
    # Create a mask for img2 (where the second image is placed)
    height2, width2 = img2.shape[:2]
    x_offset, y_offset = translation_dist

    # Create blending mask
    mask = np.zeros_like(img1[:, :, 0], dtype=np.float32)
    mask[y_offset : y_offset + height2, x_offset : x_offset + width2] = 1.0
    mask = cv2.merge([mask, mask, mask])  # Create a 3-channel mask

    # Convert images to float for blending
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    # Gaussian pyramid generation
    gp1, gp2, gp_mask = [img1], [img2], [mask]
    for _ in range(2):  # Depth of pyramid
        img1 = cv2.pyrDown(gp1[-1])
        img2 = cv2.pyrDown(gp2[-1])
        mask = cv2.pyrDown(gp_mask[-1])

        # Resize to ensure dimensions match
        min_rows = min(img1.shape[0], img2.shape[0], mask.shape[0])
        min_cols = min(img1.shape[1], img2.shape[1], mask.shape[1])
        img1 = img1[:min_rows, :min_cols]
        img2 = img2[:min_rows, :min_cols]
        mask = mask[:min_rows, :min_cols]

        gp1.append(img1)
        gp2.append(img2)
        gp_mask.append(mask)

    # Laplacian pyramid generation
    lp1 = [gp1[-1]]
    lp2 = [gp2[-1]]
    lp_mask = [gp_mask[-1]]
    for i in range(len(gp1) - 1, 0, -1):
      # Compute correct size for pyrUp (double the current Laplacian size)
      correct_size = (lp1[-1].shape[1] * 2, lp1[-1].shape[0] * 2)
      
      # Upsample and explicitly resize to match Gaussian pyramid
      upsampled_gp1 = cv2.pyrUp(lp1[-1], dstsize=correct_size)
      upsampled_gp2 = cv2.pyrUp(lp2[-1], dstsize=correct_size)
      upsampled_mask = cv2.pyrUp(lp_mask[-1], dstsize=correct_size)

      # Explicitly match dimensions with Gaussian pyramid
      upsampled_gp1 = cv2.resize(upsampled_gp1, (gp1[i - 1].shape[1], gp1[i - 1].shape[0]))
      upsampled_gp2 = cv2.resize(upsampled_gp2, (gp2[i - 1].shape[1], gp2[i - 1].shape[0]))
      upsampled_mask = cv2.resize(upsampled_mask, (gp_mask[i - 1].shape[1], gp_mask[i - 1].shape[0]))

      # Compute Laplacians and append
      lp1.append(gp1[i - 1] - upsampled_gp1)
      lp2.append(gp2[i - 1] - upsampled_gp2)
      lp_mask.append(upsampled_mask)


    # Blending pyramids
    
    blended_pyramid = []
    for l1, l2, m in zip(lp1, lp2, lp_mask):
        # Ensure all inputs have matching shapes
        target_shape = (l1.shape[1], l1.shape[0])
        l2 = cv2.resize(l2, target_shape)
        m = cv2.resize(m, target_shape)

        # Perform blending
        blended = l1 * (1 - m) + l2 * m
        blended_pyramid.append(blended)


    # Reconstructing the blended image
    blended_img = blended_pyramid[0]
    for i in range(1, len(blended_pyramid)):
        # Target size from the current pyramid level
        size = (blended_pyramid[i].shape[1], blended_pyramid[i].shape[0])
        
        # Upsample and explicitly resize to the target size
        blended_img = cv2.pyrUp(blended_img)
        blended_img = cv2.resize(blended_img, size)

        # Add the current pyramid level
        blended_img = cv2.add(blended_img, blended_pyramid[i])


    # Clip to valid range and convert to uint8
    blended_img = np.clip(blended_img, 0, 255).astype(np.uint8)
    return blended_img