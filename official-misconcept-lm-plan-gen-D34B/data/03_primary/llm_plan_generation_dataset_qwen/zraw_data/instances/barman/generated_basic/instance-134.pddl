(define (problem prob)
 (:domain barman)
 (:objects 
      shaker384 - shaker
      left right - hand
      shot481 shot462 - shot
      ingredient28 ingredient257 - ingredient
      cocktail1 - cocktail
      dispenser492 dispenser137 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker384)
  (ontable shot481)
  (ontable shot462)
  (dispenses dispenser492 ingredient28)
  (dispenses dispenser137 ingredient257)
  (clean shaker384)
  (clean shot481)
  (clean shot462)
  (empty shaker384)
  (empty shot481)
  (empty shot462)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker384 l0)
  (shaker-level shaker384 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient28)
  (cocktail-part2 cocktail1 ingredient257)
)
 (:goal
  (and
      (contains shot481 cocktail1)
)))
