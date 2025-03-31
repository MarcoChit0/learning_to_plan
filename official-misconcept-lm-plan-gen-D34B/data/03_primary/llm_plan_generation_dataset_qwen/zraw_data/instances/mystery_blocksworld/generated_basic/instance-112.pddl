

(define (problem mystery-BW-rand-4)
(:domain mystery-4ops)
(:objects a b c d )
(:init
(harmony)
(craves a c)
(planet b)
(craves c d)
(craves d b)
(province a)
)
(:goal
(and
(craves a b)
(craves d a))
)
)


