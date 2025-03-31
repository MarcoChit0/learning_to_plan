

(define (problem BW-rand-6)
(:domain blocksworld-4ops)
(:objects a b c d e f )
(:init
(handempty)
(on a d)
(on b c)
(ontable c)
(on d e)
(on e b)
(on f a)
(clear f)
)
(:goal
(and
(on a c)
(on b e)
(on c f)
(on e a))
)
)


