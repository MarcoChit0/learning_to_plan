(define (problem prob)
 (:domain barman)
 (:objects 
      shaker228 - shaker
      left right - hand
      shot299 shot488 - shot
      ingredient283 ingredient391 ingredient247 ingredient55 - ingredient
      cocktail406 - cocktail
      dispenser26 dispenser64 dispenser17 dispenser257 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker228)
  (ontable shot299)
  (ontable shot488)
  (dispenses dispenser26 ingredient283)
  (dispenses dispenser64 ingredient391)
  (dispenses dispenser17 ingredient247)
  (dispenses dispenser257 ingredient55)
  (clean shaker228)
  (clean shot299)
  (clean shot488)
  (empty shaker228)
  (empty shot299)
  (empty shot488)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker228 l0)
  (shaker-level shaker228 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail406 ingredient247)
  (cocktail-part2 cocktail406 ingredient55)
)
 (:goal
  (and
      (contains shot299 cocktail406)
)))
