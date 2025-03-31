

(define (problem BW-rand-8)
(:domain blocksworld-4ops)
(:objects a b c d e f g h )
(:init
(handempty)
(on a c)
(on b e)
(on c g)
(on d a)
(ontable e)
(on f h)
(on g b)
(on h d)
(clear f)
)
(:goal
(and
(on a e)
(on b f)
(on d h)
(on e c)
(on g b)
(on h a))
)
)


