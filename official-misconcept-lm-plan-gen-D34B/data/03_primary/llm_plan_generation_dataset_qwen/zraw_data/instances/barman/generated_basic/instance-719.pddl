(define (problem prob)
 (:domain barman)
 (:objects 
      shaker316 - shaker
      left right - hand
      shot453 - shot
      ingredient285 ingredient125 ingredient283 - ingredient
      cocktail1 - cocktail
      dispenser87 dispenser343 dispenser314 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker316)
  (ontable shot453)
  (dispenses dispenser87 ingredient285)
  (dispenses dispenser343 ingredient125)
  (dispenses dispenser314 ingredient283)
  (clean shaker316)
  (clean shot453)
  (empty shaker316)
  (empty shot453)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker316 l0)
  (shaker-level shaker316 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient285)
  (cocktail-part2 cocktail1 ingredient125)
)
 (:goal
  (and
      (contains shot453 cocktail1)
)))
