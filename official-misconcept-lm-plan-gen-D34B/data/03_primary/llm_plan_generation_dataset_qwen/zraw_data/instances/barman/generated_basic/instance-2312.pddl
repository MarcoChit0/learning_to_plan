(define (problem prob)
 (:domain barman)
 (:objects 
      shaker295 - shaker
      left right - hand
      shot274 - shot
      ingredient389 ingredient499 - ingredient
      cocktail430 - cocktail
      dispenser372 dispenser101 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker295)
  (ontable shot274)
  (dispenses dispenser372 ingredient389)
  (dispenses dispenser101 ingredient499)
  (clean shaker295)
  (clean shot274)
  (empty shaker295)
  (empty shot274)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker295 l0)
  (shaker-level shaker295 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail430 ingredient499)
  (cocktail-part2 cocktail430 ingredient389)
)
 (:goal
  (and
      (contains shot274 cocktail430)
)))
