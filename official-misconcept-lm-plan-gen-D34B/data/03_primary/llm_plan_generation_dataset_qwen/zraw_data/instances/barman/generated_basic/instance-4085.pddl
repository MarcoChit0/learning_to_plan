(define (problem prob)
 (:domain barman)
 (:objects 
      shaker120 - shaker
      left right - hand
      shot12 shot363 - shot
      ingredient213 ingredient148 ingredient189 ingredient486 - ingredient
      cocktail319 - cocktail
      dispenser456 dispenser83 dispenser228 dispenser398 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker120)
  (ontable shot12)
  (ontable shot363)
  (dispenses dispenser456 ingredient213)
  (dispenses dispenser83 ingredient148)
  (dispenses dispenser228 ingredient189)
  (dispenses dispenser398 ingredient486)
  (clean shaker120)
  (clean shot12)
  (clean shot363)
  (empty shaker120)
  (empty shot12)
  (empty shot363)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker120 l0)
  (shaker-level shaker120 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail319 ingredient148)
  (cocktail-part2 cocktail319 ingredient213)
)
 (:goal
  (and
      (contains shot12 cocktail319)
)))
