(define (problem prob)
 (:domain barman)
 (:objects 
      shaker106 - shaker
      left right - hand
      shot91 shot490 - shot
      ingredient453 ingredient32 - ingredient
      cocktail296 - cocktail
      dispenser456 dispenser179 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker106)
  (ontable shot91)
  (ontable shot490)
  (dispenses dispenser456 ingredient453)
  (dispenses dispenser179 ingredient32)
  (clean shaker106)
  (clean shot91)
  (clean shot490)
  (empty shaker106)
  (empty shot91)
  (empty shot490)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker106 l0)
  (shaker-level shaker106 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail296 ingredient32)
  (cocktail-part2 cocktail296 ingredient453)
)
 (:goal
  (and
      (contains shot91 cocktail296)
)))
