

(define (problem BW-rand-6)
(:domain blocksworld-4ops)
(:objects a b c d e f )
(:init
(handempty)
(on a d)
(on b a)
(on c b)
(ontable d)
(ontable e)
(on f c)
(clear e)
(clear f)
)
(:goal
(and
(on a f)
(on b c)
(on c d)
(on e a)
(on f b))
)
)


