(define (problem prob)
 (:domain barman)
 (:objects 
      shaker487 - shaker
      left right - hand
      shot171 shot433 shot285 - shot
      ingredient405 ingredient124 ingredient19 ingredient91 - ingredient
      cocktail1 - cocktail
      dispenser150 dispenser259 dispenser287 dispenser43 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker487)
  (ontable shot171)
  (ontable shot433)
  (ontable shot285)
  (dispenses dispenser150 ingredient405)
  (dispenses dispenser259 ingredient124)
  (dispenses dispenser287 ingredient19)
  (dispenses dispenser43 ingredient91)
  (clean shaker487)
  (clean shot171)
  (clean shot433)
  (clean shot285)
  (empty shaker487)
  (empty shot171)
  (empty shot433)
  (empty shot285)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker487 l0)
  (shaker-level shaker487 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient124)
  (cocktail-part2 cocktail1 ingredient405)
)
 (:goal
  (and
      (contains shot171 cocktail1)
      (contains shot433 cocktail1)
)))
