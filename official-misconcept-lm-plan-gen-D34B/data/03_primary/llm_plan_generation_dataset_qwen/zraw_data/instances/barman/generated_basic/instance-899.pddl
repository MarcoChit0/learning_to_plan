(define (problem prob)
 (:domain barman)
 (:objects 
      shaker494 - shaker
      left right - hand
      shot68 shot483 - shot
      ingredient170 ingredient211 - ingredient
      cocktail1 - cocktail
      dispenser113 dispenser197 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker494)
  (ontable shot68)
  (ontable shot483)
  (dispenses dispenser113 ingredient170)
  (dispenses dispenser197 ingredient211)
  (clean shaker494)
  (clean shot68)
  (clean shot483)
  (empty shaker494)
  (empty shot68)
  (empty shot483)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker494 l0)
  (shaker-level shaker494 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient211)
  (cocktail-part2 cocktail1 ingredient170)
)
 (:goal
  (and
      (contains shot68 cocktail1)
)))
