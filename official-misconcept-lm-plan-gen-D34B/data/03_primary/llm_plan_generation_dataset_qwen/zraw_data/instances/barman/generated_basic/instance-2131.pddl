(define (problem prob)
 (:domain barman)
 (:objects 
      shaker357 - shaker
      left right - hand
      shot348 shot273 shot85 - shot
      ingredient204 ingredient352 - ingredient
      cocktail269 - cocktail
      dispenser59 dispenser101 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker357)
  (ontable shot348)
  (ontable shot273)
  (ontable shot85)
  (dispenses dispenser59 ingredient204)
  (dispenses dispenser101 ingredient352)
  (clean shaker357)
  (clean shot348)
  (clean shot273)
  (clean shot85)
  (empty shaker357)
  (empty shot348)
  (empty shot273)
  (empty shot85)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker357 l0)
  (shaker-level shaker357 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail269 ingredient352)
  (cocktail-part2 cocktail269 ingredient204)
)
 (:goal
  (and
      (contains shot348 cocktail269)
      (contains shot273 ingredient352)
)))
