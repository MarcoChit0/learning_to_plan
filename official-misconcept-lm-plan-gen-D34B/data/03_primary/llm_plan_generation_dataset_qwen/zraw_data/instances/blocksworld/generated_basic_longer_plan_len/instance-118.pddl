

(define (problem BW-rand-8)
(:domain blocksworld-4ops)
(:objects a b c d e f g h )
(:init
(handempty)
(on a f)
(on b h)
(on c e)
(ontable d)
(on e a)
(on f d)
(on g b)
(on h c)
(clear g)
)
(:goal
(and
(on a f)
(on c e)
(on d b)
(on f d)
(on h c))
)
)


