(define (problem prob)
 (:domain barman)
 (:objects 
      shaker306 - shaker
      left right - hand
      shot354 - shot
      ingredient263 ingredient432 ingredient410 ingredient424 - ingredient
      cocktail432 - cocktail
      dispenser256 dispenser1 dispenser337 dispenser114 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker306)
  (ontable shot354)
  (dispenses dispenser256 ingredient263)
  (dispenses dispenser1 ingredient432)
  (dispenses dispenser337 ingredient410)
  (dispenses dispenser114 ingredient424)
  (clean shaker306)
  (clean shot354)
  (empty shaker306)
  (empty shot354)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker306 l0)
  (shaker-level shaker306 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail432 ingredient410)
  (cocktail-part2 cocktail432 ingredient424)
)
 (:goal
  (and
      (contains shot354 cocktail432)
)))
