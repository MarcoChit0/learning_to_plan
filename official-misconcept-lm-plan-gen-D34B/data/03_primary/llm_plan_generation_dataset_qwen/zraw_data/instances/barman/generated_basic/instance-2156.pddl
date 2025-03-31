(define (problem prob)
 (:domain barman)
 (:objects 
      shaker234 - shaker
      left right - hand
      shot91 - shot
      ingredient35 ingredient33 - ingredient
      cocktail428 - cocktail
      dispenser302 dispenser303 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker234)
  (ontable shot91)
  (dispenses dispenser302 ingredient35)
  (dispenses dispenser303 ingredient33)
  (clean shaker234)
  (clean shot91)
  (empty shaker234)
  (empty shot91)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker234 l0)
  (shaker-level shaker234 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail428 ingredient33)
  (cocktail-part2 cocktail428 ingredient35)
)
 (:goal
  (and
      (contains shot91 cocktail428)
)))
