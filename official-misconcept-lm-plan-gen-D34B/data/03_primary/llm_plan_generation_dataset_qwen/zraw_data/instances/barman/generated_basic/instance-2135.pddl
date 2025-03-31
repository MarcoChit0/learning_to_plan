(define (problem prob)
 (:domain barman)
 (:objects 
      shaker400 - shaker
      left right - hand
      shot250 shot182 shot159 - shot
      ingredient349 ingredient156 - ingredient
      cocktail124 - cocktail
      dispenser36 dispenser145 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker400)
  (ontable shot250)
  (ontable shot182)
  (ontable shot159)
  (dispenses dispenser36 ingredient349)
  (dispenses dispenser145 ingredient156)
  (clean shaker400)
  (clean shot250)
  (clean shot182)
  (clean shot159)
  (empty shaker400)
  (empty shot250)
  (empty shot182)
  (empty shot159)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker400 l0)
  (shaker-level shaker400 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail124 ingredient156)
  (cocktail-part2 cocktail124 ingredient349)
)
 (:goal
  (and
      (contains shot250 cocktail124)
      (contains shot182 cocktail124)
)))
