(define (problem gripper-1-3-1)
(:domain gripper-strips)
(:objects robot1 - robot
rgripper1 lgripper1 - gripper
room192 room238 room13 - room
ball253 - obj)
(:init
(at-robby robot1 room192)
(free robot1 rgripper1)
(free robot1 lgripper1)
(at ball253 room192)
)
(:goal
(and
(at ball253 room13)
)
)
)
