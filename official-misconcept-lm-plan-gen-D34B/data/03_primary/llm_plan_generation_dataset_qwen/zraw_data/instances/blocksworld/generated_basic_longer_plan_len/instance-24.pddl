

(define (problem BW-rand-6)
(:domain blocksworld-4ops)
(:objects a b c d e f )
(:init
(handempty)
(on a c)
(on b a)
(ontable c)
(ontable d)
(on e b)
(on f e)
(clear d)
(clear f)
)
(:goal
(and
(on b c)
(on c a)
(on d b)
(on e f)
(on f d))
)
)


