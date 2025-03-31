(define (problem gripper-2-2-3)
(:domain gripper-strips)
(:objects robot1 robot2 - robot
rgripper1 lgripper1 rgripper2 lgripper2 - gripper
room158 room314 - room
ball469 ball166 ball135 - obj)
(:init
(at-robby robot1 room158)
(free robot1 rgripper1)
(free robot1 lgripper1)
(at-robby robot2 room314)
(free robot2 rgripper2)
(free robot2 lgripper2)
(at ball469 room314)
(at ball166 room314)
(at ball135 room158)
)
(:goal
(and
(at ball469 room158)
(at ball166 room158)
(at ball135 room314)
)
)
)
