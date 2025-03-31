

(define (problem mystery-BW-rand-4)
(:domain mystery-4ops)
(:objects a b c d )
(:init
(harmony)
(craves a d)
(craves b a)
(craves c b)
(planet d)
(province c)
)
(:goal
(and
(craves b d)
(craves c b)
(craves d a))
)
)


