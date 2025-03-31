

(define (problem BW-rand-8)
(:domain blocksworld-4ops)
(:objects a b c d e f g h )
(:init
(handempty)
(ontable a)
(on b e)
(on c h)
(on d f)
(ontable e)
(on f a)
(on g c)
(on h d)
(clear b)
(clear g)
)
(:goal
(and
(on a d)
(on c g)
(on d e)
(on e h)
(on g a)
(on h f))
)
)


