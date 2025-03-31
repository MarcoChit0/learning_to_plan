(define (problem prob)
 (:domain barman)
 (:objects 
      shaker219 - shaker
      left right - hand
      shot202 - shot
      ingredient452 ingredient478 - ingredient
      cocktail10 - cocktail
      dispenser68 dispenser86 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker219)
  (ontable shot202)
  (dispenses dispenser68 ingredient452)
  (dispenses dispenser86 ingredient478)
  (clean shaker219)
  (clean shot202)
  (empty shaker219)
  (empty shot202)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker219 l0)
  (shaker-level shaker219 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail10 ingredient478)
  (cocktail-part2 cocktail10 ingredient452)
)
 (:goal
  (and
      (contains shot202 cocktail10)
)))
