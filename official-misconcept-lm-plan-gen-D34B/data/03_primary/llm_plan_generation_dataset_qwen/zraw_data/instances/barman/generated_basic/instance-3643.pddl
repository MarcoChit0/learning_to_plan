(define (problem prob)
 (:domain barman)
 (:objects 
      shaker292 - shaker
      left right - hand
      shot100 - shot
      ingredient368 ingredient390 ingredient460 ingredient218 - ingredient
      cocktail498 - cocktail
      dispenser306 dispenser12 dispenser309 dispenser375 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker292)
  (ontable shot100)
  (dispenses dispenser306 ingredient368)
  (dispenses dispenser12 ingredient390)
  (dispenses dispenser309 ingredient460)
  (dispenses dispenser375 ingredient218)
  (clean shaker292)
  (clean shot100)
  (empty shaker292)
  (empty shot100)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker292 l0)
  (shaker-level shaker292 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail498 ingredient368)
  (cocktail-part2 cocktail498 ingredient460)
)
 (:goal
  (and
      (contains shot100 cocktail498)
)))
