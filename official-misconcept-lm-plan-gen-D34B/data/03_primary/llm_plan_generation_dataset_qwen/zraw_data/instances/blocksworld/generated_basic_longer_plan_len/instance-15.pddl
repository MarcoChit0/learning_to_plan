

(define (problem BW-rand-6)
(:domain blocksworld-4ops)
(:objects a b c d e f )
(:init
(handempty)
(on a d)
(on b e)
(on c b)
(ontable d)
(ontable e)
(on f c)
(clear a)
(clear f)
)
(:goal
(and
(on a d)
(on c b)
(on d f)
(on e a)
(on f c))
)
)


