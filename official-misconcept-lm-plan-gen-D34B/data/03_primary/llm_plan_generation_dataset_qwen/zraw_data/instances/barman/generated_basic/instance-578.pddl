(define (problem prob)
 (:domain barman)
 (:objects 
      shaker35 - shaker
      left right - hand
      shot390 shot323 - shot
      ingredient285 ingredient262 - ingredient
      cocktail1 - cocktail
      dispenser181 dispenser341 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker35)
  (ontable shot390)
  (ontable shot323)
  (dispenses dispenser181 ingredient285)
  (dispenses dispenser341 ingredient262)
  (clean shaker35)
  (clean shot390)
  (clean shot323)
  (empty shaker35)
  (empty shot390)
  (empty shot323)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker35 l0)
  (shaker-level shaker35 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient262)
  (cocktail-part2 cocktail1 ingredient285)
)
 (:goal
  (and
      (contains shot390 cocktail1)
)))
