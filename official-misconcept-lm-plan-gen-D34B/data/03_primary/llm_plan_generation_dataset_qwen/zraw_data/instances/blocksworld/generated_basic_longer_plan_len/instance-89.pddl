

(define (problem BW-rand-8)
(:domain blocksworld-4ops)
(:objects a b c d e f g h )
(:init
(handempty)
(ontable a)
(on b a)
(on c e)
(on d c)
(on e f)
(on f b)
(on g h)
(on h d)
(clear g)
)
(:goal
(and
(on a h)
(on b c)
(on c f)
(on d b)
(on f a)
(on g e)
(on h g))
)
)


