

(define (problem BW-rand-8)
(:domain blocksworld-4ops)
(:objects a b c d e f g h )
(:init
(handempty)
(ontable a)
(on b a)
(ontable c)
(on d e)
(ontable e)
(on f c)
(on g f)
(on h b)
(clear d)
(clear g)
(clear h)
)
(:goal
(and
(on a e)
(on c b)
(on d f)
(on e c)
(on f a)
(on g d)
(on h g))
)
)


