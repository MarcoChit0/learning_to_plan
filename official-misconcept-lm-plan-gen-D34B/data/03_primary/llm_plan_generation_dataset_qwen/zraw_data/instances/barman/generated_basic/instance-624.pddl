(define (problem prob)
 (:domain barman)
 (:objects 
      shaker326 - shaker
      left right - hand
      shot466 shot106 - shot
      ingredient37 ingredient249 ingredient454 ingredient367 - ingredient
      cocktail1 - cocktail
      dispenser64 dispenser401 dispenser257 dispenser33 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker326)
  (ontable shot466)
  (ontable shot106)
  (dispenses dispenser64 ingredient37)
  (dispenses dispenser401 ingredient249)
  (dispenses dispenser257 ingredient454)
  (dispenses dispenser33 ingredient367)
  (clean shaker326)
  (clean shot466)
  (clean shot106)
  (empty shaker326)
  (empty shot466)
  (empty shot106)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker326 l0)
  (shaker-level shaker326 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient454)
  (cocktail-part2 cocktail1 ingredient249)
)
 (:goal
  (and
      (contains shot466 cocktail1)
)))
