(define (problem prob)
 (:domain barman)
 (:objects 
      shaker310 - shaker
      left right - hand
      shot131 shot6 shot70 - shot
      ingredient20 ingredient367 - ingredient
      cocktail14 - cocktail
      dispenser498 dispenser483 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker310)
  (ontable shot131)
  (ontable shot6)
  (ontable shot70)
  (dispenses dispenser498 ingredient20)
  (dispenses dispenser483 ingredient367)
  (clean shaker310)
  (clean shot131)
  (clean shot6)
  (clean shot70)
  (empty shaker310)
  (empty shot131)
  (empty shot6)
  (empty shot70)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker310 l0)
  (shaker-level shaker310 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail14 ingredient367)
  (cocktail-part2 cocktail14 ingredient20)
)
 (:goal
  (and
      (contains shot131 cocktail14)
      (contains shot6 ingredient20)
)))
