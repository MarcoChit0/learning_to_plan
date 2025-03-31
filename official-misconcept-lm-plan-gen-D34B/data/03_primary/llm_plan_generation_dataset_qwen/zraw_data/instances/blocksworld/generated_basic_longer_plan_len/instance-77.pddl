

(define (problem BW-rand-7)
(:domain blocksworld-4ops)
(:objects a b c d e f g )
(:init
(handempty)
(on a c)
(on b d)
(on c f)
(ontable d)
(on e b)
(on f g)
(on g e)
(clear a)
)
(:goal
(and
(on a c)
(on b g)
(on c f)
(on f e)
(on g d))
)
)


