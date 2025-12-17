from gqlalchemy import Memgraph
from mampfsearch.utils.models import Lecture, Course, HasLecture


class MemgraphGraphStorage:
    def __init__(self, host: str, port: int, user: str, password: str):
        self.driver = Memgraph(host=host, port=port, user=user, password=password)

    def get_lecture_node(self, name: str):
        from mampfsearch.utils.models import Lecture

        try:
            lecture = Lecture(name=name).load(self.driver)
            return lecture
        except Exception:
            return None

    def get_course_node(self, name: str):
        try:
            course = Course(name=name).load(self.driver)
            return course
        except Exception:
            return None

    def add_course_node(self, course: Course):
        course.save(self.driver)

    def add_lecture_node(self, lecture: Lecture, courseNode: Course):
        lecture.save(self.driver)
        HasLecture(_start_node_id=courseNode._id, _end_node_id=lecture._id).save(
            self.driver
        )
