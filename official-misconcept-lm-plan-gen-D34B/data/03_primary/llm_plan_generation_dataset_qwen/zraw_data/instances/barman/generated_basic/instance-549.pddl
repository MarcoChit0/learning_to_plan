(define (problem prob)
 (:domain barman)
 (:objects 
      shaker80 - shaker
      left right - hand
      shot325 - shot
      ingredient492 ingredient318 ingredient458 - ingredient
      cocktail1 - cocktail
      dispenser294 dispenser271 dispenser430 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker80)
  (ontable shot325)
  (dispenses dispenser294 ingredient492)
  (dispenses dispenser271 ingredient318)
  (dispenses dispenser430 ingredient458)
  (clean shaker80)
  (clean shot325)
  (empty shaker80)
  (empty shot325)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker80 l0)
  (shaker-level shaker80 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient318)
  (cocktail-part2 cocktail1 ingredient458)
)
 (:goal
  (and
      (contains shot325 cocktail1)
)))
