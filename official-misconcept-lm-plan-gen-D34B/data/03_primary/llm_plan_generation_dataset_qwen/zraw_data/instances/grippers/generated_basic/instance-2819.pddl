(define (problem gripper-2-2-3)
(:domain gripper-strips)
(:objects robot1 robot2 - robot
rgripper1 lgripper1 rgripper2 lgripper2 - gripper
room84 room267 - room
ball52 ball361 ball295 - obj)
(:init
(at-robby robot1 room267)
(free robot1 rgripper1)
(free robot1 lgripper1)
(at-robby robot2 room84)
(free robot2 rgripper2)
(free robot2 lgripper2)
(at ball52 room267)
(at ball361 room84)
(at ball295 room267)
)
(:goal
(and
(at ball52 room84)
(at ball361 room84)
(at ball295 room267)
)
)
)
