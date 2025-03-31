

(define (problem BW-rand-8)
(:domain blocksworld-4ops)
(:objects a b c d e f g h )
(:init
(handempty)
(on a e)
(on b a)
(on c b)
(on d f)
(ontable e)
(on f c)
(on g h)
(on h d)
(clear g)
)
(:goal
(and
(on a b)
(on b f)
(on c d)
(on d a)
(on e h)
(on g e)
(on h c))
)
)


