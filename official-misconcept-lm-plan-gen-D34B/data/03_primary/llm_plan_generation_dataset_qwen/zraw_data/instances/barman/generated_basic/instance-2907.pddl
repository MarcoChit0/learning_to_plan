(define (problem prob)
 (:domain barman)
 (:objects 
      shaker257 - shaker
      left right - hand
      shot160 - shot
      ingredient442 ingredient229 - ingredient
      cocktail134 - cocktail
      dispenser249 dispenser194 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker257)
  (ontable shot160)
  (dispenses dispenser249 ingredient442)
  (dispenses dispenser194 ingredient229)
  (clean shaker257)
  (clean shot160)
  (empty shaker257)
  (empty shot160)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker257 l0)
  (shaker-level shaker257 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail134 ingredient229)
  (cocktail-part2 cocktail134 ingredient442)
)
 (:goal
  (and
      (contains shot160 cocktail134)
)))
