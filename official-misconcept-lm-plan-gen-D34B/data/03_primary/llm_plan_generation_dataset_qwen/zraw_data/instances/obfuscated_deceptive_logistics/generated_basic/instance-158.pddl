


(define (problem obfuscated_deceptive_logistics-c2-s1-p1-a3)
(:domain obfuscated_deceptive_logistics-strips)
(:objects a0 a1 a2 
          c0 c1 
          t0 t1 
          l0-0 l1-0 
          p0 
)
(:init
    (cats a0)
    (cats a1)
    (cats a2)
    (stupendous c0)
    (stupendous c1)
    (sneeze t0)
    (sneeze t1)
    (texture l0-0)
    (collect  l0-0 c0)
    (texture l1-0)
    (collect  l1-0 c1)
    (spring l0-0)
    (spring l1-0)
    (hand p0)
    (next t0 l0-0)
    (next t1 l1-0)
    (next p0 l1-0)
    (next a0 l1-0)
    (next a1 l0-0)
    (next a2 l0-0)
)
(:goal
    (and
        (next p0 l0-0)
    )
)
)


