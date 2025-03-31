(define (problem prob)
 (:domain barman)
 (:objects 
      shaker63 - shaker
      left right - hand
      shot172 shot403 shot127 - shot
      ingredient258 ingredient360 - ingredient
      cocktail354 - cocktail
      dispenser224 dispenser120 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker63)
  (ontable shot172)
  (ontable shot403)
  (ontable shot127)
  (dispenses dispenser224 ingredient258)
  (dispenses dispenser120 ingredient360)
  (clean shaker63)
  (clean shot172)
  (clean shot403)
  (clean shot127)
  (empty shaker63)
  (empty shot172)
  (empty shot403)
  (empty shot127)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker63 l0)
  (shaker-level shaker63 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail354 ingredient360)
  (cocktail-part2 cocktail354 ingredient258)
)
 (:goal
  (and
      (contains shot172 cocktail354)
      (contains shot403 cocktail354)
)))
