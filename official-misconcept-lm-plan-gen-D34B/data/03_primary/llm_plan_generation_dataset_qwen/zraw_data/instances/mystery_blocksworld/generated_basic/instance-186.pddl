

(define (problem mystery-BW-rand-4)
(:domain mystery-4ops)
(:objects a b c d )
(:init
(harmony)
(craves a c)
(craves b a)
(planet c)
(craves d b)
(province d)
)
(:goal
(and
(craves b a)
(craves c b)
(craves d c))
)
)


