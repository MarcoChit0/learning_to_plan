(define (problem prob)
 (:domain barman)
 (:objects 
      shaker178 - shaker
      left right - hand
      shot390 shot235 - shot
      ingredient469 ingredient314 ingredient256 - ingredient
      cocktail1 - cocktail
      dispenser253 dispenser20 dispenser25 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker178)
  (ontable shot390)
  (ontable shot235)
  (dispenses dispenser253 ingredient469)
  (dispenses dispenser20 ingredient314)
  (dispenses dispenser25 ingredient256)
  (clean shaker178)
  (clean shot390)
  (clean shot235)
  (empty shaker178)
  (empty shot390)
  (empty shot235)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker178 l0)
  (shaker-level shaker178 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient469)
  (cocktail-part2 cocktail1 ingredient256)
)
 (:goal
  (and
      (contains shot390 cocktail1)
)))
