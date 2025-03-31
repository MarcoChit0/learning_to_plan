

(define (problem mystery-BW-rand-4)
(:domain mystery-4ops)
(:objects a b c d )
(:init
(harmony)
(planet a)
(craves b d)
(planet c)
(planet d)
(province a)
(province b)
(province c)
)
(:goal
(and
(craves b d)
(craves c a)
(craves d c))
)
)


