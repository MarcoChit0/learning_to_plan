(define (problem prob)
 (:domain barman)
 (:objects 
      shaker122 - shaker
      left right - hand
      shot464 shot346 - shot
      ingredient230 ingredient44 - ingredient
      cocktail322 - cocktail
      dispenser32 dispenser402 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker122)
  (ontable shot464)
  (ontable shot346)
  (dispenses dispenser32 ingredient230)
  (dispenses dispenser402 ingredient44)
  (clean shaker122)
  (clean shot464)
  (clean shot346)
  (empty shaker122)
  (empty shot464)
  (empty shot346)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker122 l0)
  (shaker-level shaker122 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail322 ingredient44)
  (cocktail-part2 cocktail322 ingredient230)
)
 (:goal
  (and
      (contains shot464 cocktail322)
)))
