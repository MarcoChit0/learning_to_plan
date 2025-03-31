(define (problem gripper-1-2-1)
(:domain gripper-strips)
(:objects robot1 - robot
rgripper1 lgripper1 - gripper
room148 room162 - room
ball386 - obj)
(:init
(at-robby robot1 room148)
(free robot1 rgripper1)
(free robot1 lgripper1)
(at ball386 room148)
)
(:goal
(and
(at ball386 room162)
)
)
)
