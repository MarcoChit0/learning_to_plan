(define (problem prob)
 (:domain barman)
 (:objects 
      shaker266 - shaker
      left right - hand
      shot397 shot277 - shot
      ingredient378 ingredient277 ingredient128 ingredient180 - ingredient
      cocktail1 - cocktail
      dispenser69 dispenser220 dispenser438 dispenser203 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker266)
  (ontable shot397)
  (ontable shot277)
  (dispenses dispenser69 ingredient378)
  (dispenses dispenser220 ingredient277)
  (dispenses dispenser438 ingredient128)
  (dispenses dispenser203 ingredient180)
  (clean shaker266)
  (clean shot397)
  (clean shot277)
  (empty shaker266)
  (empty shot397)
  (empty shot277)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker266 l0)
  (shaker-level shaker266 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient128)
  (cocktail-part2 cocktail1 ingredient277)
)
 (:goal
  (and
      (contains shot397 cocktail1)
)))
