

(define (problem BW-rand-6)
(:domain blocksworld-4ops)
(:objects a b c d e f )
(:init
(handempty)
(on a e)
(on b c)
(ontable c)
(on d a)
(on e b)
(on f d)
(clear f)
)
(:goal
(and
(on a d)
(on c b)
(on d e)
(on f a))
)
)


