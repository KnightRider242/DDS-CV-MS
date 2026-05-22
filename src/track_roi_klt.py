def get_face_rois(face_box):
    x, y, w, h = face_box

    rois = {
        "left_eye": (
            x + int(0.10 * w),
            y + int(0.22 * h),
            int(0.35 * w),
            int(0.22 * h),
        ),
        "right_eye": (
            x + int(0.55 * w),
            y + int(0.22 * h),
            int(0.35 * w),
            int(0.22 * h),
        ),
        "mouth": (
            x + int(0.25 * w),
            y + int(0.62 * h),
            int(0.50 * w),
            int(0.25 * h),
        ),
        "head": (
            x,
            y,
            w,
            h,
        ),
    }

    return rois