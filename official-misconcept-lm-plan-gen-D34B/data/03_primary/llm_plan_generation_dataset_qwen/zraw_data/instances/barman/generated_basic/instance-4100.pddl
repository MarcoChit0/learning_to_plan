(define (problem prob)
 (:domain barman)
 (:objects 
      shaker300 - shaker
      left right - hand
      shot330 shot178 - shot
      ingredient440 ingredient268 - ingredient
      cocktail239 - cocktail
      dispenser382 dispenser397 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker300)
  (ontable shot330)
  (ontable shot178)
  (dispenses dispenser382 ingredient440)
  (dispenses dispenser397 ingredient268)
  (clean shaker300)
  (clean shot330)
  (clean shot178)
  (empty shaker300)
  (empty shot330)
  (empty shot178)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker300 l0)
  (shaker-level shaker300 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail239 ingredient268)
  (cocktail-part2 cocktail239 ingredient440)
)
 (:goal
  (and
      (contains shot330 cocktail239)
)))
