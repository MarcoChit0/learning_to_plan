(define (problem prob)
 (:domain barman)
 (:objects 
      shaker472 - shaker
      left right - hand
      shot328 - shot
      ingredient33 ingredient75 ingredient336 - ingredient
      cocktail433 - cocktail
      dispenser463 dispenser5 dispenser49 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker472)
  (ontable shot328)
  (dispenses dispenser463 ingredient33)
  (dispenses dispenser5 ingredient75)
  (dispenses dispenser49 ingredient336)
  (clean shaker472)
  (clean shot328)
  (empty shaker472)
  (empty shot328)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker472 l0)
  (shaker-level shaker472 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail433 ingredient75)
  (cocktail-part2 cocktail433 ingredient336)
)
 (:goal
  (and
      (contains shot328 cocktail433)
)))
