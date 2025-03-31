

(define (problem BW-rand-6)
(:domain blocksworld-4ops)
(:objects a b c d e f )
(:init
(handempty)
(on a e)
(ontable b)
(on c a)
(on d f)
(ontable e)
(on f c)
(clear b)
(clear d)
)
(:goal
(and
(on a c)
(on b e)
(on c d)
(on d b)
(on e f))
)
)


