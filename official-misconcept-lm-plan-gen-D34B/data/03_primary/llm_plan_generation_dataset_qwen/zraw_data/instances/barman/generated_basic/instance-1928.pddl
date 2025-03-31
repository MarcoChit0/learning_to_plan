(define (problem prob)
 (:domain barman)
 (:objects 
      shaker397 - shaker
      left right - hand
      shot226 - shot
      ingredient321 ingredient14 - ingredient
      cocktail221 - cocktail
      dispenser264 dispenser307 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker397)
  (ontable shot226)
  (dispenses dispenser264 ingredient321)
  (dispenses dispenser307 ingredient14)
  (clean shaker397)
  (clean shot226)
  (empty shaker397)
  (empty shot226)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker397 l0)
  (shaker-level shaker397 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail221 ingredient321)
  (cocktail-part2 cocktail221 ingredient14)
)
 (:goal
  (and
      (contains shot226 cocktail221)
)))
