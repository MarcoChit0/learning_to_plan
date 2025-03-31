

(define (problem BW-rand-8)
(:domain blocksworld-4ops)
(:objects a b c d e f g h )
(:init
(handempty)
(ontable a)
(on b a)
(on c g)
(on d f)
(ontable e)
(ontable f)
(on g e)
(on h d)
(clear b)
(clear c)
(clear h)
)
(:goal
(and
(on a f)
(on b a)
(on c d)
(on d g)
(on e b)
(on g e)
(on h c))
)
)


