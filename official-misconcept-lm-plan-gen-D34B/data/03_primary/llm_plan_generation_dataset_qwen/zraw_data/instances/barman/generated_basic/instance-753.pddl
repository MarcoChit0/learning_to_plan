(define (problem prob)
 (:domain barman)
 (:objects 
      shaker143 - shaker
      left right - hand
      shot176 shot57 - shot
      ingredient370 ingredient454 ingredient63 - ingredient
      cocktail1 - cocktail
      dispenser127 dispenser301 dispenser162 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker143)
  (ontable shot176)
  (ontable shot57)
  (dispenses dispenser127 ingredient370)
  (dispenses dispenser301 ingredient454)
  (dispenses dispenser162 ingredient63)
  (clean shaker143)
  (clean shot176)
  (clean shot57)
  (empty shaker143)
  (empty shot176)
  (empty shot57)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker143 l0)
  (shaker-level shaker143 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient454)
  (cocktail-part2 cocktail1 ingredient63)
)
 (:goal
  (and
      (contains shot176 cocktail1)
)))
