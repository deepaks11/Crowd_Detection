from shapely.geometry import Polygon, Point


class PolygonTest:

    def __init__(self, detections, line_zones):
        self.detections = detections
        self.zones = line_zones
        self.polygon = Polygon([self.zones[0], self.zones[1], self.zones[2], self.zones[3], self.zones[4]])

    def point_polygon_test(self):
        try:
            count = 0
            any_inside_or_touching = False  # To track if any object is inside or touching the polygon

            for xyxy in self.detections.xyxy:
                p_x1, p_y1, p_x2, p_y2 = xyxy.astype(int)
                person_coord = [(p_x1, p_y1), (p_x1, p_y2), (p_x2, p_y1), (p_x2, p_y2)]

                # Check if any corner of the bounding box is inside the polygon
                inside = False
                for coord in person_coord:
                    point = Point(coord)
                    if self.polygon.contains(point):
                        inside = True
                        break  # Exit the loop if the point is inside

                # Check if the bounding box intersects with the polygon
                rect_polygon = Polygon(person_coord)
                if inside or self.polygon.intersects(rect_polygon):
                    count += 1
                    any_inside_or_touching = True

            return any_inside_or_touching, count

        except Exception as ex:
            print(ex)
            return False, 0  # Return False and 0 if an exception occurs
