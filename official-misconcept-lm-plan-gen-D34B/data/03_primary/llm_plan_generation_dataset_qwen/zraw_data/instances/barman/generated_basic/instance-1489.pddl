(define (problem prob)
 (:domain barman)
 (:objects 
      shaker295 - shaker
      left right - hand
      shot418 shot76 - shot
      ingredient306 ingredient133 ingredient151 ingredient130 - ingredient
      cocktail397 - cocktail
      dispenser447 dispenser123 dispenser198 dispenser162 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker295)
  (ontable shot418)
  (ontable shot76)
  (dispenses dispenser447 ingredient306)
  (dispenses dispenser123 ingredient133)
  (dispenses dispenser198 ingredient151)
  (dispenses dispenser162 ingredient130)
  (clean shaker295)
  (clean shot418)
  (clean shot76)
  (empty shaker295)
  (empty shot418)
  (empty shot76)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker295 l0)
  (shaker-level shaker295 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail397 ingredient306)
  (cocktail-part2 cocktail397 ingredient130)
)
 (:goal
  (and
      (contains shot418 cocktail397)
)))
