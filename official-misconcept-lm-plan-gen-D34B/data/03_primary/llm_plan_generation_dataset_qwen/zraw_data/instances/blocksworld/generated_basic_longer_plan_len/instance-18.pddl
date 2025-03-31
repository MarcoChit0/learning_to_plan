

(define (problem BW-rand-6)
(:domain blocksworld-4ops)
(:objects a b c d e f )
(:init
(handempty)
(on a c)
(on b a)
(ontable c)
(on d e)
(on e b)
(ontable f)
(clear d)
(clear f)
)
(:goal
(and
(on b d)
(on c b)
(on d e)
(on e f)
(on f a))
)
)


