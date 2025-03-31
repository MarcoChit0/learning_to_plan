(define (problem prob)
 (:domain barman)
 (:objects 
      shaker153 - shaker
      left right - hand
      shot24 shot427 shot71 - shot
      ingredient50 ingredient437 ingredient431 ingredient353 - ingredient
      cocktail165 - cocktail
      dispenser192 dispenser256 dispenser334 dispenser339 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker153)
  (ontable shot24)
  (ontable shot427)
  (ontable shot71)
  (dispenses dispenser192 ingredient50)
  (dispenses dispenser256 ingredient437)
  (dispenses dispenser334 ingredient431)
  (dispenses dispenser339 ingredient353)
  (clean shaker153)
  (clean shot24)
  (clean shot427)
  (clean shot71)
  (empty shaker153)
  (empty shot24)
  (empty shot427)
  (empty shot71)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker153 l0)
  (shaker-level shaker153 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail165 ingredient431)
  (cocktail-part2 cocktail165 ingredient353)
)
 (:goal
  (and
      (contains shot24 cocktail165)
      (contains shot427 ingredient431)
)))
