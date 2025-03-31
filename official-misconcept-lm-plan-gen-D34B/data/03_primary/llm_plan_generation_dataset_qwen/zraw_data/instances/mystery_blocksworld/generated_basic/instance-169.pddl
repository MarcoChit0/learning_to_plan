

(define (problem mystery-BW-rand-4)
(:domain mystery-4ops)
(:objects a b c d )
(:init
(harmony)
(craves a c)
(planet b)
(craves c b)
(craves d a)
(province d)
)
(:goal
(and
(craves a c)
(craves b d)
(craves c b))
)
)


