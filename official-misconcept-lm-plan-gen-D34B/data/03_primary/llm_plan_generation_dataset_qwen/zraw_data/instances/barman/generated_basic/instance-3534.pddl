(define (problem prob)
 (:domain barman)
 (:objects 
      shaker110 - shaker
      left right - hand
      shot321 shot120 - shot
      ingredient417 ingredient496 ingredient430 ingredient24 - ingredient
      cocktail10 - cocktail
      dispenser236 dispenser435 dispenser38 dispenser275 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker110)
  (ontable shot321)
  (ontable shot120)
  (dispenses dispenser236 ingredient417)
  (dispenses dispenser435 ingredient496)
  (dispenses dispenser38 ingredient430)
  (dispenses dispenser275 ingredient24)
  (clean shaker110)
  (clean shot321)
  (clean shot120)
  (empty shaker110)
  (empty shot321)
  (empty shot120)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker110 l0)
  (shaker-level shaker110 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail10 ingredient417)
  (cocktail-part2 cocktail10 ingredient496)
)
 (:goal
  (and
      (contains shot321 cocktail10)
)))
