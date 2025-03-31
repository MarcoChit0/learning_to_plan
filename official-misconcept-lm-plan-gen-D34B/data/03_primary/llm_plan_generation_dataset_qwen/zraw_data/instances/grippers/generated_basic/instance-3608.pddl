(define (problem gripper-1-6-2)
(:domain gripper-strips)
(:objects robot1 - robot
rgripper1 lgripper1 - gripper
room237 room254 room81 room96 room8 room486 - room
ball352 ball337 - obj)
(:init
(at-robby robot1 room254)
(free robot1 rgripper1)
(free robot1 lgripper1)
(at ball352 room96)
(at ball337 room96)
)
(:goal
(and
(at ball352 room254)
(at ball337 room8)
)
)
)
