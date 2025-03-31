(define (problem gripper-1-3-3)
(:domain gripper-strips)
(:objects robot1 - robot
rgripper1 lgripper1 - gripper
room421 room324 room459 - room
ball358 ball176 ball423 - obj)
(:init
(at-robby robot1 room324)
(free robot1 rgripper1)
(free robot1 lgripper1)
(at ball358 room459)
(at ball176 room459)
(at ball423 room459)
)
(:goal
(and
(at ball358 room459)
(at ball176 room459)
(at ball423 room421)
)
)
)
