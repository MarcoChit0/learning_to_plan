(define (problem prob)
 (:domain barman)
 (:objects 
      shaker81 - shaker
      left right - hand
      shot63 - shot
      ingredient348 ingredient188 ingredient483 ingredient356 - ingredient
      cocktail337 - cocktail
      dispenser440 dispenser476 dispenser333 dispenser276 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker81)
  (ontable shot63)
  (dispenses dispenser440 ingredient348)
  (dispenses dispenser476 ingredient188)
  (dispenses dispenser333 ingredient483)
  (dispenses dispenser276 ingredient356)
  (clean shaker81)
  (clean shot63)
  (empty shaker81)
  (empty shot63)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker81 l0)
  (shaker-level shaker81 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail337 ingredient188)
  (cocktail-part2 cocktail337 ingredient348)
)
 (:goal
  (and
      (contains shot63 cocktail337)
)))
