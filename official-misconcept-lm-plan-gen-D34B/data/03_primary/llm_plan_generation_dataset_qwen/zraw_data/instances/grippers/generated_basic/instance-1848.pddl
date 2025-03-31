(define (problem gripper-4-2-2)
(:domain gripper-strips)
(:objects robot1 robot2 robot3 robot4 - robot
rgripper1 lgripper1 rgripper2 lgripper2 rgripper3 lgripper3 rgripper4 lgripper4 - gripper
room298 room296 - room
ball149 ball316 - obj)
(:init
(at-robby robot1 room298)
(free robot1 rgripper1)
(free robot1 lgripper1)
(at-robby robot2 room296)
(free robot2 rgripper2)
(free robot2 lgripper2)
(at-robby robot3 room298)
(free robot3 rgripper3)
(free robot3 lgripper3)
(at-robby robot4 room296)
(free robot4 rgripper4)
(free robot4 lgripper4)
(at ball149 room298)
(at ball316 room296)
)
(:goal
(and
(at ball149 room296)
(at ball316 room298)
)
)
)
