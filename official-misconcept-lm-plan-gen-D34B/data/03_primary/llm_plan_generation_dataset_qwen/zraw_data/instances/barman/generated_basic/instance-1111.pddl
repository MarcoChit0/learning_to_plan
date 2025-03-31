(define (problem prob)
 (:domain barman)
 (:objects 
      shaker91 - shaker
      left right - hand
      shot190 shot222 - shot
      ingredient73 ingredient88 - ingredient
      cocktail1 - cocktail
      dispenser267 dispenser114 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker91)
  (ontable shot190)
  (ontable shot222)
  (dispenses dispenser267 ingredient73)
  (dispenses dispenser114 ingredient88)
  (clean shaker91)
  (clean shot190)
  (clean shot222)
  (empty shaker91)
  (empty shot190)
  (empty shot222)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker91 l0)
  (shaker-level shaker91 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient73)
  (cocktail-part2 cocktail1 ingredient88)
)
 (:goal
  (and
      (contains shot190 cocktail1)
)))
