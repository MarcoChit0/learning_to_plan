(define (problem prob)
 (:domain barman)
 (:objects 
      shaker489 - shaker
      left right - hand
      shot162 shot197 - shot
      ingredient399 ingredient487 - ingredient
      cocktail101 - cocktail
      dispenser86 dispenser48 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker489)
  (ontable shot162)
  (ontable shot197)
  (dispenses dispenser86 ingredient399)
  (dispenses dispenser48 ingredient487)
  (clean shaker489)
  (clean shot162)
  (clean shot197)
  (empty shaker489)
  (empty shot162)
  (empty shot197)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker489 l0)
  (shaker-level shaker489 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail101 ingredient487)
  (cocktail-part2 cocktail101 ingredient399)
)
 (:goal
  (and
      (contains shot162 cocktail101)
)))
