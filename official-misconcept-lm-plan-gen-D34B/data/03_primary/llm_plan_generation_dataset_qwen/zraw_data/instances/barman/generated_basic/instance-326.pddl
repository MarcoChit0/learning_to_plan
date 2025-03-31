(define (problem prob)
 (:domain barman)
 (:objects 
      shaker126 - shaker
      left right - hand
      shot154 - shot
      ingredient370 ingredient147 - ingredient
      cocktail1 - cocktail
      dispenser73 dispenser343 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker126)
  (ontable shot154)
  (dispenses dispenser73 ingredient370)
  (dispenses dispenser343 ingredient147)
  (clean shaker126)
  (clean shot154)
  (empty shaker126)
  (empty shot154)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker126 l0)
  (shaker-level shaker126 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient147)
  (cocktail-part2 cocktail1 ingredient370)
)
 (:goal
  (and
      (contains shot154 cocktail1)
)))
