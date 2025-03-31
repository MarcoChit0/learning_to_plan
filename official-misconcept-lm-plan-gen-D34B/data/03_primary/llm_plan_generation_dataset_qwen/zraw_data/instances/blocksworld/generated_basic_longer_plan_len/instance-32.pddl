

(define (problem BW-rand-6)
(:domain blocksworld-4ops)
(:objects a b c d e f )
(:init
(handempty)
(on a d)
(on b a)
(on c b)
(on d e)
(ontable e)
(on f c)
(clear f)
)
(:goal
(and
(on a f)
(on b c)
(on c e)
(on e d)
(on f b))
)
)


