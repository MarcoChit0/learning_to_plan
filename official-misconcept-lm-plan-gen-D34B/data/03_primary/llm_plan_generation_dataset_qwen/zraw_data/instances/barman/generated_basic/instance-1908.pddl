(define (problem prob)
 (:domain barman)
 (:objects 
      shaker403 - shaker
      left right - hand
      shot426 - shot
      ingredient397 ingredient366 ingredient308 ingredient68 - ingredient
      cocktail330 - cocktail
      dispenser419 dispenser114 dispenser256 dispenser112 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker403)
  (ontable shot426)
  (dispenses dispenser419 ingredient397)
  (dispenses dispenser114 ingredient366)
  (dispenses dispenser256 ingredient308)
  (dispenses dispenser112 ingredient68)
  (clean shaker403)
  (clean shot426)
  (empty shaker403)
  (empty shot426)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker403 l0)
  (shaker-level shaker403 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail330 ingredient366)
  (cocktail-part2 cocktail330 ingredient397)
)
 (:goal
  (and
      (contains shot426 cocktail330)
)))
