

(define (problem BW-rand-8)
(:domain blocksworld-4ops)
(:objects a b c d e f g h )
(:init
(handempty)
(on a e)
(on b h)
(on c b)
(on d a)
(on e f)
(on f c)
(ontable g)
(on h g)
(clear d)
)
(:goal
(and
(on a e)
(on b c)
(on d a)
(on e h)
(on f d)
(on g b)
(on h g))
)
)


