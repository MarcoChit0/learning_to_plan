(define (problem prob)
 (:domain barman)
 (:objects 
      shaker495 - shaker
      left right - hand
      shot143 - shot
      ingredient68 ingredient221 - ingredient
      cocktail1 - cocktail
      dispenser426 dispenser192 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker495)
  (ontable shot143)
  (dispenses dispenser426 ingredient68)
  (dispenses dispenser192 ingredient221)
  (clean shaker495)
  (clean shot143)
  (empty shaker495)
  (empty shot143)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker495 l0)
  (shaker-level shaker495 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient221)
  (cocktail-part2 cocktail1 ingredient68)
)
 (:goal
  (and
      (contains shot143 cocktail1)
)))
