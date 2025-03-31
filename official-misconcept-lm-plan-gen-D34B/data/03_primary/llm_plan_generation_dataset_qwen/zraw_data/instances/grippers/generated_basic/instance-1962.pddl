(define (problem gripper-4-2-2)
(:domain gripper-strips)
(:objects robot1 robot2 robot3 robot4 - robot
rgripper1 lgripper1 rgripper2 lgripper2 rgripper3 lgripper3 rgripper4 lgripper4 - gripper
room141 room364 - room
ball26 ball409 - obj)
(:init
(at-robby robot1 room364)
(free robot1 rgripper1)
(free robot1 lgripper1)
(at-robby robot2 room364)
(free robot2 rgripper2)
(free robot2 lgripper2)
(at-robby robot3 room141)
(free robot3 rgripper3)
(free robot3 lgripper3)
(at-robby robot4 room364)
(free robot4 rgripper4)
(free robot4 lgripper4)
(at ball26 room364)
(at ball409 room364)
)
(:goal
(and
(at ball26 room364)
(at ball409 room141)
)
)
)
