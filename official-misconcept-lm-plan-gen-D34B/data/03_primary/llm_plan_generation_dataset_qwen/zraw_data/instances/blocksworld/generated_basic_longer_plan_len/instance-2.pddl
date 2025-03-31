

(define (problem BW-rand-6)
(:domain blocksworld-4ops)
(:objects a b c d e f )
(:init
(handempty)
(on a c)
(ontable b)
(on c d)
(on d b)
(on e a)
(on f e)
(clear f)
)
(:goal
(and
(on c b)
(on d e)
(on e f)
(on f c))
)
)


