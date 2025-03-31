

(define (problem mystery-BW-rand-4)
(:domain mystery-4ops)
(:objects a b c d )
(:init
(harmony)
(craves a c)
(planet b)
(craves c d)
(planet d)
(province a)
(province b)
)
(:goal
(and
(craves a b)
(craves b d)
(craves d c))
)
)


