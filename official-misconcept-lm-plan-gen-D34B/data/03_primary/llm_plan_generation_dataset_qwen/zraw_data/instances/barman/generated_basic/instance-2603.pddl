(define (problem prob)
 (:domain barman)
 (:objects 
      shaker0 - shaker
      left right - hand
      shot395 shot414 - shot
      ingredient123 ingredient170 ingredient475 - ingredient
      cocktail43 - cocktail
      dispenser361 dispenser273 dispenser322 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker0)
  (ontable shot395)
  (ontable shot414)
  (dispenses dispenser361 ingredient123)
  (dispenses dispenser273 ingredient170)
  (dispenses dispenser322 ingredient475)
  (clean shaker0)
  (clean shot395)
  (clean shot414)
  (empty shaker0)
  (empty shot395)
  (empty shot414)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker0 l0)
  (shaker-level shaker0 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail43 ingredient475)
  (cocktail-part2 cocktail43 ingredient170)
)
 (:goal
  (and
      (contains shot395 cocktail43)
)))
