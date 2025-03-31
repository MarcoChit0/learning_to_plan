(define (problem gripper-1-2-4)
(:domain gripper-strips)
(:objects robot1 - robot
rgripper1 lgripper1 - gripper
room373 room398 - room
ball323 ball386 ball28 ball125 - obj)
(:init
(at-robby robot1 room398)
(free robot1 rgripper1)
(free robot1 lgripper1)
(at ball323 room398)
(at ball386 room373)
(at ball28 room373)
(at ball125 room373)
)
(:goal
(and
(at ball323 room373)
(at ball386 room398)
(at ball28 room373)
(at ball125 room398)
)
)
)
