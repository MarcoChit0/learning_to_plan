(define (problem gripper-1-3-1)
(:domain gripper-strips)
(:objects robot1 - robot
rgripper1 lgripper1 - gripper
room318 room494 room148 - room
ball451 - obj)
(:init
(at-robby robot1 room318)
(free robot1 rgripper1)
(free robot1 lgripper1)
(at ball451 room148)
)
(:goal
(and
(at ball451 room494)
)
)
)
