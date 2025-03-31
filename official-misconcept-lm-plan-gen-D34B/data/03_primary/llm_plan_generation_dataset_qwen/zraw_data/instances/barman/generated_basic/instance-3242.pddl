(define (problem prob)
 (:domain barman)
 (:objects 
      shaker407 - shaker
      left right - hand
      shot82 - shot
      ingredient292 ingredient118 ingredient478 - ingredient
      cocktail243 - cocktail
      dispenser189 dispenser48 dispenser444 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker407)
  (ontable shot82)
  (dispenses dispenser189 ingredient292)
  (dispenses dispenser48 ingredient118)
  (dispenses dispenser444 ingredient478)
  (clean shaker407)
  (clean shot82)
  (empty shaker407)
  (empty shot82)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker407 l0)
  (shaker-level shaker407 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail243 ingredient478)
  (cocktail-part2 cocktail243 ingredient292)
)
 (:goal
  (and
      (contains shot82 cocktail243)
)))
