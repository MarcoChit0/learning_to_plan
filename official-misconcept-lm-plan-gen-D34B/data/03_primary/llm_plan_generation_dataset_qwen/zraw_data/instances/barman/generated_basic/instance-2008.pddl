(define (problem prob)
 (:domain barman)
 (:objects 
      shaker178 - shaker
      left right - hand
      shot326 shot6 shot243 - shot
      ingredient399 ingredient410 ingredient214 - ingredient
      cocktail167 - cocktail
      dispenser12 dispenser85 dispenser478 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker178)
  (ontable shot326)
  (ontable shot6)
  (ontable shot243)
  (dispenses dispenser12 ingredient399)
  (dispenses dispenser85 ingredient410)
  (dispenses dispenser478 ingredient214)
  (clean shaker178)
  (clean shot326)
  (clean shot6)
  (clean shot243)
  (empty shaker178)
  (empty shot326)
  (empty shot6)
  (empty shot243)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker178 l0)
  (shaker-level shaker178 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail167 ingredient399)
  (cocktail-part2 cocktail167 ingredient214)
)
 (:goal
  (and
      (contains shot326 cocktail167)
      (contains shot6 ingredient214)
)))
