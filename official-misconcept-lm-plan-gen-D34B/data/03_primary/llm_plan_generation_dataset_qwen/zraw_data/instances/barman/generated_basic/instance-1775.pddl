(define (problem prob)
 (:domain barman)
 (:objects 
      shaker72 - shaker
      left right - hand
      shot196 shot302 - shot
      ingredient106 ingredient389 ingredient145 - ingredient
      cocktail55 - cocktail
      dispenser384 dispenser20 dispenser84 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker72)
  (ontable shot196)
  (ontable shot302)
  (dispenses dispenser384 ingredient106)
  (dispenses dispenser20 ingredient389)
  (dispenses dispenser84 ingredient145)
  (clean shaker72)
  (clean shot196)
  (clean shot302)
  (empty shaker72)
  (empty shot196)
  (empty shot302)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker72 l0)
  (shaker-level shaker72 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail55 ingredient389)
  (cocktail-part2 cocktail55 ingredient106)
)
 (:goal
  (and
      (contains shot196 cocktail55)
)))
