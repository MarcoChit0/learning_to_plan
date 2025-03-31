(define (problem prob)
 (:domain barman)
 (:objects 
      shaker340 - shaker
      left right - hand
      shot449 - shot
      ingredient491 ingredient25 ingredient283 ingredient74 - ingredient
      cocktail185 - cocktail
      dispenser408 dispenser126 dispenser386 dispenser243 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker340)
  (ontable shot449)
  (dispenses dispenser408 ingredient491)
  (dispenses dispenser126 ingredient25)
  (dispenses dispenser386 ingredient283)
  (dispenses dispenser243 ingredient74)
  (clean shaker340)
  (clean shot449)
  (empty shaker340)
  (empty shot449)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker340 l0)
  (shaker-level shaker340 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail185 ingredient74)
  (cocktail-part2 cocktail185 ingredient25)
)
 (:goal
  (and
      (contains shot449 cocktail185)
)))
