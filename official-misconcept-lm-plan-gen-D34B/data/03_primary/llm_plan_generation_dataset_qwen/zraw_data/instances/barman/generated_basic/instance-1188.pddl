(define (problem prob)
 (:domain barman)
 (:objects 
      shaker495 - shaker
      left right - hand
      shot249 shot125 - shot
      ingredient300 ingredient258 - ingredient
      cocktail5 - cocktail
      dispenser394 dispenser277 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker495)
  (ontable shot249)
  (ontable shot125)
  (dispenses dispenser394 ingredient300)
  (dispenses dispenser277 ingredient258)
  (clean shaker495)
  (clean shot249)
  (clean shot125)
  (empty shaker495)
  (empty shot249)
  (empty shot125)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker495 l0)
  (shaker-level shaker495 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail5 ingredient258)
  (cocktail-part2 cocktail5 ingredient300)
)
 (:goal
  (and
      (contains shot249 cocktail5)
)))
