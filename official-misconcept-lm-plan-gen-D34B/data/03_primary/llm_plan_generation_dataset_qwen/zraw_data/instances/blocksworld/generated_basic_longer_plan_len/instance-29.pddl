

(define (problem BW-rand-6)
(:domain blocksworld-4ops)
(:objects a b c d e f )
(:init
(handempty)
(ontable a)
(on b e)
(on c b)
(on d c)
(on e a)
(ontable f)
(clear d)
(clear f)
)
(:goal
(and
(on a d)
(on b c)
(on c a)
(on d f)
(on f e))
)
)


