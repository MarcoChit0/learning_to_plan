

(define (problem BW-rand-6)
(:domain blocksworld-4ops)
(:objects a b c d e f )
(:init
(handempty)
(on a e)
(ontable b)
(on c f)
(on d b)
(on e c)
(on f d)
(clear a)
)
(:goal
(and
(on a d)
(on b a)
(on c e)
(on e f)
(on f b))
)
)


