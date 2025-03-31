(define (problem prob)
 (:domain barman)
 (:objects 
      shaker6 - shaker
      left right - hand
      shot487 shot436 - shot
      ingredient74 ingredient23 ingredient293 ingredient367 - ingredient
      cocktail337 - cocktail
      dispenser22 dispenser292 dispenser404 dispenser372 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker6)
  (ontable shot487)
  (ontable shot436)
  (dispenses dispenser22 ingredient74)
  (dispenses dispenser292 ingredient23)
  (dispenses dispenser404 ingredient293)
  (dispenses dispenser372 ingredient367)
  (clean shaker6)
  (clean shot487)
  (clean shot436)
  (empty shaker6)
  (empty shot487)
  (empty shot436)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker6 l0)
  (shaker-level shaker6 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail337 ingredient293)
  (cocktail-part2 cocktail337 ingredient23)
)
 (:goal
  (and
      (contains shot487 cocktail337)
)))
