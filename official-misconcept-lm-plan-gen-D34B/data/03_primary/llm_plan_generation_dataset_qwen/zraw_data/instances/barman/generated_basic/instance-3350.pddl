(define (problem prob)
 (:domain barman)
 (:objects 
      shaker76 - shaker
      left right - hand
      shot196 - shot
      ingredient456 ingredient277 - ingredient
      cocktail54 - cocktail
      dispenser55 dispenser361 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker76)
  (ontable shot196)
  (dispenses dispenser55 ingredient456)
  (dispenses dispenser361 ingredient277)
  (clean shaker76)
  (clean shot196)
  (empty shaker76)
  (empty shot196)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker76 l0)
  (shaker-level shaker76 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail54 ingredient277)
  (cocktail-part2 cocktail54 ingredient456)
)
 (:goal
  (and
      (contains shot196 cocktail54)
)))
