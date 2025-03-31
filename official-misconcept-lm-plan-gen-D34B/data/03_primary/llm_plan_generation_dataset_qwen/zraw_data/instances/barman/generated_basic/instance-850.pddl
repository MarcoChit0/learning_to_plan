(define (problem prob)
 (:domain barman)
 (:objects 
      shaker397 - shaker
      left right - hand
      shot229 - shot
      ingredient257 ingredient240 ingredient226 - ingredient
      cocktail1 - cocktail
      dispenser97 dispenser63 dispenser370 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker397)
  (ontable shot229)
  (dispenses dispenser97 ingredient257)
  (dispenses dispenser63 ingredient240)
  (dispenses dispenser370 ingredient226)
  (clean shaker397)
  (clean shot229)
  (empty shaker397)
  (empty shot229)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker397 l0)
  (shaker-level shaker397 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient226)
  (cocktail-part2 cocktail1 ingredient240)
)
 (:goal
  (and
      (contains shot229 cocktail1)
)))
