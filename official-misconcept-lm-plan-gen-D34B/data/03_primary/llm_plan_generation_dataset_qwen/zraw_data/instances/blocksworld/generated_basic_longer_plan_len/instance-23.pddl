

(define (problem BW-rand-6)
(:domain blocksworld-4ops)
(:objects a b c d e f )
(:init
(handempty)
(ontable a)
(on b a)
(on c d)
(on d e)
(on e f)
(on f b)
(clear c)
)
(:goal
(and
(on b d)
(on c b)
(on d f)
(on f a))
)
)


