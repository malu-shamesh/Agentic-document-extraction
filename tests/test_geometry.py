from src.geometry import bbox_from_polygon, filter_and_sort_regions


def test_bbox_from_polygon_returns_min_max_extent():
    polygon = [[57, 3], [336, 0], [336, 39], [58, 44]]

    bbox = bbox_from_polygon(polygon)

    assert bbox == [57, 0, 336, 44]


def test_filter_and_sort_regions_drops_low_confidence_and_sorts_descending():
    regions = [
        {"label": "text", "score": 0.9, "bbox": [0, 0, 1, 1]},
        {"label": "chart", "score": 0.3, "bbox": [0, 0, 1, 1]},
        {"label": "table", "score": 0.7, "bbox": [0, 0, 1, 1]},
    ]

    result = filter_and_sort_regions(regions, min_confidence=0.5)

    assert [r["label"] for r in result] == ["text", "table"]
