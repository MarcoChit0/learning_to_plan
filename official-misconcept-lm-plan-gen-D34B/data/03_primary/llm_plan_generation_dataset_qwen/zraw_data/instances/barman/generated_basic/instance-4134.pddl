(define (problem prob)
 (:domain barman)
 (:objects 
      shaker399 - shaker
      left right - hand
      shot389 shot308 - shot
      ingredient340 ingredient12 ingredient403 - ingredient
      cocktail262 - cocktail
      dispenser362 dispenser224 dispenser252 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker399)
  (ontable shot389)
  (ontable shot308)
  (dispenses dispenser362 ingredient340)
  (dispenses dispenser224 ingredient12)
  (dispenses dispenser252 ingredient403)
  (clean shaker399)
  (clean shot389)
  (clean shot308)
  (empty shaker399)
  (empty shot389)
  (empty shot308)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker399 l0)
  (shaker-level shaker399 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail262 ingredient12)
  (cocktail-part2 cocktail262 ingredient403)
)
 (:goal
  (and
      (contains shot389 cocktail262)
)))
