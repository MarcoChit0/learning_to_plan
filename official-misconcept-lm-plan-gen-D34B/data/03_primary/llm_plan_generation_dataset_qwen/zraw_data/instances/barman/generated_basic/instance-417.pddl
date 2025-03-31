(define (problem prob)
 (:domain barman)
 (:objects 
      shaker280 - shaker
      left right - hand
      shot66 - shot
      ingredient266 ingredient229 - ingredient
      cocktail1 - cocktail
      dispenser255 dispenser306 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker280)
  (ontable shot66)
  (dispenses dispenser255 ingredient266)
  (dispenses dispenser306 ingredient229)
  (clean shaker280)
  (clean shot66)
  (empty shaker280)
  (empty shot66)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker280 l0)
  (shaker-level shaker280 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient266)
  (cocktail-part2 cocktail1 ingredient229)
)
 (:goal
  (and
      (contains shot66 cocktail1)
)))
