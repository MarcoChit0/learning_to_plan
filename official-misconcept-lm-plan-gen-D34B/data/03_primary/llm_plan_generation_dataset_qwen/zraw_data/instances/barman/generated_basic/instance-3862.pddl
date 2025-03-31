(define (problem prob)
 (:domain barman)
 (:objects 
      shaker469 - shaker
      left right - hand
      shot172 shot378 - shot
      ingredient450 ingredient204 ingredient410 - ingredient
      cocktail482 - cocktail
      dispenser295 dispenser45 dispenser99 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker469)
  (ontable shot172)
  (ontable shot378)
  (dispenses dispenser295 ingredient450)
  (dispenses dispenser45 ingredient204)
  (dispenses dispenser99 ingredient410)
  (clean shaker469)
  (clean shot172)
  (clean shot378)
  (empty shaker469)
  (empty shot172)
  (empty shot378)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker469 l0)
  (shaker-level shaker469 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail482 ingredient410)
  (cocktail-part2 cocktail482 ingredient204)
)
 (:goal
  (and
      (contains shot172 cocktail482)
)))
