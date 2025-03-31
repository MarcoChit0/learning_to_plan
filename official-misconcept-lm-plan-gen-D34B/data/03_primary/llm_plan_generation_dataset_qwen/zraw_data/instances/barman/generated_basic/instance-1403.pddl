(define (problem prob)
 (:domain barman)
 (:objects 
      shaker78 - shaker
      left right - hand
      shot293 - shot
      ingredient494 ingredient102 ingredient300 ingredient268 - ingredient
      cocktail274 - cocktail
      dispenser468 dispenser101 dispenser95 dispenser205 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker78)
  (ontable shot293)
  (dispenses dispenser468 ingredient494)
  (dispenses dispenser101 ingredient102)
  (dispenses dispenser95 ingredient300)
  (dispenses dispenser205 ingredient268)
  (clean shaker78)
  (clean shot293)
  (empty shaker78)
  (empty shot293)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker78 l0)
  (shaker-level shaker78 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail274 ingredient102)
  (cocktail-part2 cocktail274 ingredient494)
)
 (:goal
  (and
      (contains shot293 cocktail274)
)))
