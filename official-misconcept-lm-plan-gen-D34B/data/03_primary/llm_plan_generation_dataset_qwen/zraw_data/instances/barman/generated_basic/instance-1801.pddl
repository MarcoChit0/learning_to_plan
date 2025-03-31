(define (problem prob)
 (:domain barman)
 (:objects 
      shaker453 - shaker
      left right - hand
      shot125 shot156 - shot
      ingredient10 ingredient42 ingredient125 ingredient149 - ingredient
      cocktail379 - cocktail
      dispenser83 dispenser395 dispenser133 dispenser484 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker453)
  (ontable shot125)
  (ontable shot156)
  (dispenses dispenser83 ingredient10)
  (dispenses dispenser395 ingredient42)
  (dispenses dispenser133 ingredient125)
  (dispenses dispenser484 ingredient149)
  (clean shaker453)
  (clean shot125)
  (clean shot156)
  (empty shaker453)
  (empty shot125)
  (empty shot156)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker453 l0)
  (shaker-level shaker453 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail379 ingredient125)
  (cocktail-part2 cocktail379 ingredient42)
)
 (:goal
  (and
      (contains shot125 cocktail379)
)))
