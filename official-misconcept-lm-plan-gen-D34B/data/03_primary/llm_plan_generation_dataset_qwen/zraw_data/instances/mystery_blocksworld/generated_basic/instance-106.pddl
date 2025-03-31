

(define (problem mystery-BW-rand-4)
(:domain mystery-4ops)
(:objects a b c d )
(:init
(harmony)
(planet a)
(craves b c)
(craves c d)
(planet d)
(province a)
(province b)
)
(:goal
(and
(craves a c)
(craves b d)
(craves c b))
)
)


