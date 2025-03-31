(define (problem prob)
 (:domain barman)
 (:objects 
      shaker102 - shaker
      left right - hand
      shot408 - shot
      ingredient256 ingredient453 ingredient362 ingredient427 - ingredient
      cocktail308 - cocktail
      dispenser16 dispenser203 dispenser257 dispenser45 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker102)
  (ontable shot408)
  (dispenses dispenser16 ingredient256)
  (dispenses dispenser203 ingredient453)
  (dispenses dispenser257 ingredient362)
  (dispenses dispenser45 ingredient427)
  (clean shaker102)
  (clean shot408)
  (empty shaker102)
  (empty shot408)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker102 l0)
  (shaker-level shaker102 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail308 ingredient362)
  (cocktail-part2 cocktail308 ingredient453)
)
 (:goal
  (and
      (contains shot408 cocktail308)
)))
