

(define (problem BW-rand-8)
(:domain blocksworld-4ops)
(:objects a b c d e f g h )
(:init
(handempty)
(ontable a)
(on b h)
(on c g)
(on d e)
(on e b)
(ontable f)
(on g f)
(on h c)
(clear a)
(clear d)
)
(:goal
(and
(on a b)
(on b f)
(on c e)
(on d c)
(on e a)
(on h d))
)
)


