(define (problem prob)
 (:domain barman)
 (:objects 
      shaker209 - shaker
      left right - hand
      shot173 shot488 - shot
      ingredient163 ingredient302 - ingredient
      cocktail348 - cocktail
      dispenser378 dispenser462 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker209)
  (ontable shot173)
  (ontable shot488)
  (dispenses dispenser378 ingredient163)
  (dispenses dispenser462 ingredient302)
  (clean shaker209)
  (clean shot173)
  (clean shot488)
  (empty shaker209)
  (empty shot173)
  (empty shot488)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker209 l0)
  (shaker-level shaker209 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail348 ingredient163)
  (cocktail-part2 cocktail348 ingredient302)
)
 (:goal
  (and
      (contains shot173 cocktail348)
)))
