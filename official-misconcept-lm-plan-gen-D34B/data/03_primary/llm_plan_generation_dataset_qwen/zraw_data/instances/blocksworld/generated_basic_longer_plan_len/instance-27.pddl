

(define (problem BW-rand-6)
(:domain blocksworld-4ops)
(:objects a b c d e f )
(:init
(handempty)
(ontable a)
(on b d)
(on c b)
(on d a)
(ontable e)
(on f c)
(clear e)
(clear f)
)
(:goal
(and
(on a e)
(on b d)
(on c b)
(on d a)
(on f c))
)
)


