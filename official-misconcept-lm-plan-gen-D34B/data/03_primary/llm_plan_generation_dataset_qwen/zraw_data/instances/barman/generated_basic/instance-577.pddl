(define (problem prob)
 (:domain barman)
 (:objects 
      shaker486 - shaker
      left right - hand
      shot135 shot353 - shot
      ingredient309 ingredient425 - ingredient
      cocktail1 - cocktail
      dispenser214 dispenser191 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker486)
  (ontable shot135)
  (ontable shot353)
  (dispenses dispenser214 ingredient309)
  (dispenses dispenser191 ingredient425)
  (clean shaker486)
  (clean shot135)
  (clean shot353)
  (empty shaker486)
  (empty shot135)
  (empty shot353)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker486 l0)
  (shaker-level shaker486 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient425)
  (cocktail-part2 cocktail1 ingredient309)
)
 (:goal
  (and
      (contains shot135 cocktail1)
)))
