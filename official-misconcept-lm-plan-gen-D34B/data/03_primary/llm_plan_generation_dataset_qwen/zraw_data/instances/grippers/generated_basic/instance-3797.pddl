(define (problem gripper-1-3-2)
(:domain gripper-strips)
(:objects robot1 - robot
rgripper1 lgripper1 - gripper
room377 room465 room210 - room
ball498 ball173 - obj)
(:init
(at-robby robot1 room377)
(free robot1 rgripper1)
(free robot1 lgripper1)
(at ball498 room377)
(at ball173 room465)
)
(:goal
(and
(at ball498 room465)
(at ball173 room377)
)
)
)
