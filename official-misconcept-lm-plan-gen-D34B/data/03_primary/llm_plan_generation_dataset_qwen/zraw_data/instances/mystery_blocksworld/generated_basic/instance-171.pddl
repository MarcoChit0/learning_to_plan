

(define (problem mystery-BW-rand-4)
(:domain mystery-4ops)
(:objects a b c d )
(:init
(harmony)
(craves a d)
(craves b c)
(planet c)
(craves d b)
(province a)
)
(:goal
(and
(craves a b)
(craves c a)
(craves d c))
)
)


