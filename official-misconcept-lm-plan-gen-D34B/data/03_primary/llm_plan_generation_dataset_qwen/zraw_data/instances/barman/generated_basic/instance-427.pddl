(define (problem prob)
 (:domain barman)
 (:objects 
      shaker69 - shaker
      left right - hand
      shot265 - shot
      ingredient372 ingredient426 - ingredient
      cocktail1 - cocktail
      dispenser303 dispenser356 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker69)
  (ontable shot265)
  (dispenses dispenser303 ingredient372)
  (dispenses dispenser356 ingredient426)
  (clean shaker69)
  (clean shot265)
  (empty shaker69)
  (empty shot265)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker69 l0)
  (shaker-level shaker69 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient372)
  (cocktail-part2 cocktail1 ingredient426)
)
 (:goal
  (and
      (contains shot265 cocktail1)
)))
