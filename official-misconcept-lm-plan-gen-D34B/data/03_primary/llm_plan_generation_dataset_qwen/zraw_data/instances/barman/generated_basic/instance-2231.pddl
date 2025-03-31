(define (problem prob)
 (:domain barman)
 (:objects 
      shaker243 - shaker
      left right - hand
      shot328 - shot
      ingredient315 ingredient283 - ingredient
      cocktail60 - cocktail
      dispenser399 dispenser240 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker243)
  (ontable shot328)
  (dispenses dispenser399 ingredient315)
  (dispenses dispenser240 ingredient283)
  (clean shaker243)
  (clean shot328)
  (empty shaker243)
  (empty shot328)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker243 l0)
  (shaker-level shaker243 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail60 ingredient315)
  (cocktail-part2 cocktail60 ingredient283)
)
 (:goal
  (and
      (contains shot328 cocktail60)
)))
