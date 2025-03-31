(define (problem prob)
 (:domain barman)
 (:objects 
      shaker306 - shaker
      left right - hand
      shot438 - shot
      ingredient308 ingredient311 ingredient381 ingredient492 - ingredient
      cocktail4 - cocktail
      dispenser294 dispenser339 dispenser146 dispenser275 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker306)
  (ontable shot438)
  (dispenses dispenser294 ingredient308)
  (dispenses dispenser339 ingredient311)
  (dispenses dispenser146 ingredient381)
  (dispenses dispenser275 ingredient492)
  (clean shaker306)
  (clean shot438)
  (empty shaker306)
  (empty shot438)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker306 l0)
  (shaker-level shaker306 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail4 ingredient492)
  (cocktail-part2 cocktail4 ingredient308)
)
 (:goal
  (and
      (contains shot438 cocktail4)
)))
