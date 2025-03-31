

(define (problem BW-rand-6)
(:domain blocksworld-4ops)
(:objects a b c d e f )
(:init
(handempty)
(on a d)
(on b e)
(on c a)
(on d b)
(ontable e)
(on f c)
(clear f)
)
(:goal
(and
(on a d)
(on b c)
(on c f)
(on d b)
(on f e))
)
)


