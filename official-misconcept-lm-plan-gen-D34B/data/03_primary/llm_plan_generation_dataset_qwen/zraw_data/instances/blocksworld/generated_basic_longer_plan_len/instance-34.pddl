

(define (problem BW-rand-6)
(:domain blocksworld-4ops)
(:objects a b c d e f )
(:init
(handempty)
(on a f)
(on b e)
(on c a)
(on d c)
(ontable e)
(on f b)
(clear d)
)
(:goal
(and
(on a d)
(on c a)
(on d b)
(on f e))
)
)


