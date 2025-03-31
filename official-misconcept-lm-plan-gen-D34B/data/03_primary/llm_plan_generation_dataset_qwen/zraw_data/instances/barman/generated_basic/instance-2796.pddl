(define (problem prob)
 (:domain barman)
 (:objects 
      shaker310 - shaker
      left right - hand
      shot407 shot210 - shot
      ingredient137 ingredient179 - ingredient
      cocktail494 - cocktail
      dispenser266 dispenser111 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker310)
  (ontable shot407)
  (ontable shot210)
  (dispenses dispenser266 ingredient137)
  (dispenses dispenser111 ingredient179)
  (clean shaker310)
  (clean shot407)
  (clean shot210)
  (empty shaker310)
  (empty shot407)
  (empty shot210)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker310 l0)
  (shaker-level shaker310 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail494 ingredient179)
  (cocktail-part2 cocktail494 ingredient137)
)
 (:goal
  (and
      (contains shot407 cocktail494)
)))
