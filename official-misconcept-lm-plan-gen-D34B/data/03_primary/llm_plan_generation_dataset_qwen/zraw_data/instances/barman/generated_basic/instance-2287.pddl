(define (problem prob)
 (:domain barman)
 (:objects 
      shaker89 - shaker
      left right - hand
      shot16 - shot
      ingredient75 ingredient214 ingredient395 - ingredient
      cocktail100 - cocktail
      dispenser367 dispenser139 dispenser433 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker89)
  (ontable shot16)
  (dispenses dispenser367 ingredient75)
  (dispenses dispenser139 ingredient214)
  (dispenses dispenser433 ingredient395)
  (clean shaker89)
  (clean shot16)
  (empty shaker89)
  (empty shot16)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker89 l0)
  (shaker-level shaker89 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail100 ingredient395)
  (cocktail-part2 cocktail100 ingredient214)
)
 (:goal
  (and
      (contains shot16 cocktail100)
)))
