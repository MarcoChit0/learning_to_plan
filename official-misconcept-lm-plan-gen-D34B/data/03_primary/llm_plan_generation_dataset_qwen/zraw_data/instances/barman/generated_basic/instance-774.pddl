(define (problem prob)
 (:domain barman)
 (:objects 
      shaker327 - shaker
      left right - hand
      shot389 - shot
      ingredient93 ingredient27 - ingredient
      cocktail1 - cocktail
      dispenser38 dispenser397 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker327)
  (ontable shot389)
  (dispenses dispenser38 ingredient93)
  (dispenses dispenser397 ingredient27)
  (clean shaker327)
  (clean shot389)
  (empty shaker327)
  (empty shot389)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker327 l0)
  (shaker-level shaker327 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient27)
  (cocktail-part2 cocktail1 ingredient93)
)
 (:goal
  (and
      (contains shot389 cocktail1)
)))
