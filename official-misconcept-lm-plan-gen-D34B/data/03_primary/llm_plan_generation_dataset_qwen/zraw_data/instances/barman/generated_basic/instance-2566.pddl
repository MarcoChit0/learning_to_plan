(define (problem prob)
 (:domain barman)
 (:objects 
      shaker386 - shaker
      left right - hand
      shot379 shot131 shot203 - shot
      ingredient234 ingredient1 - ingredient
      cocktail126 - cocktail
      dispenser402 dispenser111 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker386)
  (ontable shot379)
  (ontable shot131)
  (ontable shot203)
  (dispenses dispenser402 ingredient234)
  (dispenses dispenser111 ingredient1)
  (clean shaker386)
  (clean shot379)
  (clean shot131)
  (clean shot203)
  (empty shaker386)
  (empty shot379)
  (empty shot131)
  (empty shot203)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker386 l0)
  (shaker-level shaker386 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail126 ingredient1)
  (cocktail-part2 cocktail126 ingredient234)
)
 (:goal
  (and
      (contains shot379 cocktail126)
      (contains shot131 cocktail126)
)))
