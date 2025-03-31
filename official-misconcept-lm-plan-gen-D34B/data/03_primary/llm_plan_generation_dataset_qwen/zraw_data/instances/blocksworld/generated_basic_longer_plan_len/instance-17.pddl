

(define (problem BW-rand-6)
(:domain blocksworld-4ops)
(:objects a b c d e f )
(:init
(handempty)
(on a e)
(on b f)
(on c a)
(ontable d)
(on e b)
(ontable f)
(clear c)
(clear d)
)
(:goal
(and
(on a c)
(on c b)
(on d f)
(on e d)
(on f a))
)
)


