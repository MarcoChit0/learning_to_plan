(define (problem prob)
 (:domain barman)
 (:objects 
      shaker230 - shaker
      left right - hand
      shot125 shot381 - shot
      ingredient197 ingredient260 ingredient437 - ingredient
      cocktail74 - cocktail
      dispenser474 dispenser2 dispenser5 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker230)
  (ontable shot125)
  (ontable shot381)
  (dispenses dispenser474 ingredient197)
  (dispenses dispenser2 ingredient260)
  (dispenses dispenser5 ingredient437)
  (clean shaker230)
  (clean shot125)
  (clean shot381)
  (empty shaker230)
  (empty shot125)
  (empty shot381)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker230 l0)
  (shaker-level shaker230 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail74 ingredient260)
  (cocktail-part2 cocktail74 ingredient437)
)
 (:goal
  (and
      (contains shot125 cocktail74)
)))
