(define (problem prob)
 (:domain barman)
 (:objects 
      shaker402 - shaker
      left right - hand
      shot326 shot83 - shot
      ingredient234 ingredient455 ingredient240 - ingredient
      cocktail408 - cocktail
      dispenser48 dispenser158 dispenser101 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker402)
  (ontable shot326)
  (ontable shot83)
  (dispenses dispenser48 ingredient234)
  (dispenses dispenser158 ingredient455)
  (dispenses dispenser101 ingredient240)
  (clean shaker402)
  (clean shot326)
  (clean shot83)
  (empty shaker402)
  (empty shot326)
  (empty shot83)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker402 l0)
  (shaker-level shaker402 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail408 ingredient455)
  (cocktail-part2 cocktail408 ingredient240)
)
 (:goal
  (and
      (contains shot326 cocktail408)
)))
