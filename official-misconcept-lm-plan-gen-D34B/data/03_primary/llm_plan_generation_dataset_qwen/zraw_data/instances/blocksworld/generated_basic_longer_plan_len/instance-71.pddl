

(define (problem BW-rand-7)
(:domain blocksworld-4ops)
(:objects a b c d e f g )
(:init
(handempty)
(on a f)
(on b e)
(on c g)
(ontable d)
(on e d)
(on f b)
(ontable g)
(clear a)
(clear c)
)
(:goal
(and
(on a d)
(on b e)
(on c f)
(on e a)
(on f g)
(on g b))
)
)


