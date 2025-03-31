(define (problem prob)
 (:domain barman)
 (:objects 
      shaker499 - shaker
      left right - hand
      shot9 shot102 shot418 - shot
      ingredient72 ingredient104 - ingredient
      cocktail276 - cocktail
      dispenser498 dispenser108 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker499)
  (ontable shot9)
  (ontable shot102)
  (ontable shot418)
  (dispenses dispenser498 ingredient72)
  (dispenses dispenser108 ingredient104)
  (clean shaker499)
  (clean shot9)
  (clean shot102)
  (clean shot418)
  (empty shaker499)
  (empty shot9)
  (empty shot102)
  (empty shot418)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker499 l0)
  (shaker-level shaker499 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail276 ingredient104)
  (cocktail-part2 cocktail276 ingredient72)
)
 (:goal
  (and
      (contains shot9 cocktail276)
      (contains shot102 ingredient104)
)))
