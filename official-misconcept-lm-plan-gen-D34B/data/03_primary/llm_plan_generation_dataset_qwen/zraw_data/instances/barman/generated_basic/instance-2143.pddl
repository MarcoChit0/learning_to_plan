(define (problem prob)
 (:domain barman)
 (:objects 
      shaker420 - shaker
      left right - hand
      shot230 - shot
      ingredient466 ingredient240 - ingredient
      cocktail213 - cocktail
      dispenser401 dispenser279 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker420)
  (ontable shot230)
  (dispenses dispenser401 ingredient466)
  (dispenses dispenser279 ingredient240)
  (clean shaker420)
  (clean shot230)
  (empty shaker420)
  (empty shot230)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker420 l0)
  (shaker-level shaker420 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail213 ingredient466)
  (cocktail-part2 cocktail213 ingredient240)
)
 (:goal
  (and
      (contains shot230 cocktail213)
)))
