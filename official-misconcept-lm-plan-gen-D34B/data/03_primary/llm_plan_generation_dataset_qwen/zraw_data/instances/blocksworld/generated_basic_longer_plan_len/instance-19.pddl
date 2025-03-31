

(define (problem BW-rand-6)
(:domain blocksworld-4ops)
(:objects a b c d e f )
(:init
(handempty)
(ontable a)
(on b e)
(on c d)
(on d f)
(on e c)
(on f a)
(clear b)
)
(:goal
(and
(on a d)
(on b c)
(on c e)
(on e f))
)
)


