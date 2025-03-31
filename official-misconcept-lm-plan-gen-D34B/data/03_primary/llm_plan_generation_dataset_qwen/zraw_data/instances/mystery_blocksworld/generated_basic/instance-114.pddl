

(define (problem mystery-BW-rand-4)
(:domain mystery-4ops)
(:objects a b c d )
(:init
(harmony)
(craves a d)
(craves b c)
(planet c)
(planet d)
(province a)
(province b)
)
(:goal
(and
(craves b d)
(craves d c))
)
)


