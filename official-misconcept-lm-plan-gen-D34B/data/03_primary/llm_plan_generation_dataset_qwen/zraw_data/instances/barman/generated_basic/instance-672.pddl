(define (problem prob)
 (:domain barman)
 (:objects 
      shaker304 - shaker
      left right - hand
      shot352 shot459 - shot
      ingredient196 ingredient455 - ingredient
      cocktail1 - cocktail
      dispenser350 dispenser359 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker304)
  (ontable shot352)
  (ontable shot459)
  (dispenses dispenser350 ingredient196)
  (dispenses dispenser359 ingredient455)
  (clean shaker304)
  (clean shot352)
  (clean shot459)
  (empty shaker304)
  (empty shot352)
  (empty shot459)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker304 l0)
  (shaker-level shaker304 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient455)
  (cocktail-part2 cocktail1 ingredient196)
)
 (:goal
  (and
      (contains shot352 cocktail1)
)))
