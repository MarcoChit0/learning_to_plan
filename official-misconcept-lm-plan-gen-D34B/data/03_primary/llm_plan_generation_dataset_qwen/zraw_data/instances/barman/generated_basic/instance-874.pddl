(define (problem prob)
 (:domain barman)
 (:objects 
      shaker224 - shaker
      left right - hand
      shot334 - shot
      ingredient213 ingredient204 - ingredient
      cocktail1 - cocktail
      dispenser22 dispenser41 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker224)
  (ontable shot334)
  (dispenses dispenser22 ingredient213)
  (dispenses dispenser41 ingredient204)
  (clean shaker224)
  (clean shot334)
  (empty shaker224)
  (empty shot334)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker224 l0)
  (shaker-level shaker224 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient204)
  (cocktail-part2 cocktail1 ingredient213)
)
 (:goal
  (and
      (contains shot334 cocktail1)
)))
