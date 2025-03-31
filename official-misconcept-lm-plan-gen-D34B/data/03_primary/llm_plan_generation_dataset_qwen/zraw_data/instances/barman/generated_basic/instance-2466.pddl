(define (problem prob)
 (:domain barman)
 (:objects 
      shaker37 - shaker
      left right - hand
      shot243 - shot
      ingredient57 ingredient251 ingredient194 - ingredient
      cocktail374 - cocktail
      dispenser296 dispenser301 dispenser397 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker37)
  (ontable shot243)
  (dispenses dispenser296 ingredient57)
  (dispenses dispenser301 ingredient251)
  (dispenses dispenser397 ingredient194)
  (clean shaker37)
  (clean shot243)
  (empty shaker37)
  (empty shot243)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker37 l0)
  (shaker-level shaker37 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail374 ingredient194)
  (cocktail-part2 cocktail374 ingredient251)
)
 (:goal
  (and
      (contains shot243 cocktail374)
)))
