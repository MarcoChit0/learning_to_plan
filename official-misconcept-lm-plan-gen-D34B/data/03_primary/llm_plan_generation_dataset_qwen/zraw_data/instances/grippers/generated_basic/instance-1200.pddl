(define (problem gripper-1-3-1)
(:domain gripper-strips)
(:objects robot1 - robot
rgripper1 lgripper1 - gripper
room306 room404 room322 - room
ball150 - obj)
(:init
(at-robby robot1 room306)
(free robot1 rgripper1)
(free robot1 lgripper1)
(at ball150 room322)
)
(:goal
(and
(at ball150 room404)
)
)
)
