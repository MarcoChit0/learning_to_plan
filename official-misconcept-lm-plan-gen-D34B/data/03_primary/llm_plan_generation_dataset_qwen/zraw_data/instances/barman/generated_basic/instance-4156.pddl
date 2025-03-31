(define (problem prob)
 (:domain barman)
 (:objects 
      shaker4 - shaker
      left right - hand
      shot194 shot350 - shot
      ingredient307 ingredient380 - ingredient
      cocktail349 - cocktail
      dispenser249 dispenser6 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker4)
  (ontable shot194)
  (ontable shot350)
  (dispenses dispenser249 ingredient307)
  (dispenses dispenser6 ingredient380)
  (clean shaker4)
  (clean shot194)
  (clean shot350)
  (empty shaker4)
  (empty shot194)
  (empty shot350)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker4 l0)
  (shaker-level shaker4 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail349 ingredient380)
  (cocktail-part2 cocktail349 ingredient307)
)
 (:goal
  (and
      (contains shot194 cocktail349)
)))
