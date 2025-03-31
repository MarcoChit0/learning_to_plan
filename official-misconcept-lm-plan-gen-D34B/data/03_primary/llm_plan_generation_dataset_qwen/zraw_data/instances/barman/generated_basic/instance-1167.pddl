(define (problem prob)
 (:domain barman)
 (:objects 
      shaker245 - shaker
      left right - hand
      shot326 shot109 - shot
      ingredient257 ingredient470 ingredient426 - ingredient
      cocktail1 - cocktail
      dispenser408 dispenser366 dispenser354 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker245)
  (ontable shot326)
  (ontable shot109)
  (dispenses dispenser408 ingredient257)
  (dispenses dispenser366 ingredient470)
  (dispenses dispenser354 ingredient426)
  (clean shaker245)
  (clean shot326)
  (clean shot109)
  (empty shaker245)
  (empty shot326)
  (empty shot109)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker245 l0)
  (shaker-level shaker245 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient257)
  (cocktail-part2 cocktail1 ingredient426)
)
 (:goal
  (and
      (contains shot326 cocktail1)
)))
