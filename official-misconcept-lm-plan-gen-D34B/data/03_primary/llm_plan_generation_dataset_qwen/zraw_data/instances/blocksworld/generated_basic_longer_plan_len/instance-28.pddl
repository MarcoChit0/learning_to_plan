

(define (problem BW-rand-6)
(:domain blocksworld-4ops)
(:objects a b c d e f )
(:init
(handempty)
(on a b)
(on b f)
(on c a)
(on d e)
(on e c)
(ontable f)
(clear d)
)
(:goal
(and
(on a d)
(on b e)
(on c a)
(on d b)
(on f c))
)
)


