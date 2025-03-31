(define (problem prob)
 (:domain barman)
 (:objects 
      shaker457 - shaker
      left right - hand
      shot308 shot472 - shot
      ingredient48 ingredient295 - ingredient
      cocktail1 - cocktail
      dispenser448 dispenser350 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker457)
  (ontable shot308)
  (ontable shot472)
  (dispenses dispenser448 ingredient48)
  (dispenses dispenser350 ingredient295)
  (clean shaker457)
  (clean shot308)
  (clean shot472)
  (empty shaker457)
  (empty shot308)
  (empty shot472)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker457 l0)
  (shaker-level shaker457 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient48)
  (cocktail-part2 cocktail1 ingredient295)
)
 (:goal
  (and
      (contains shot308 cocktail1)
)))
