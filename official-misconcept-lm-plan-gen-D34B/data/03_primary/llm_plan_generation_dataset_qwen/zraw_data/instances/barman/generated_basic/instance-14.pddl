(define (problem prob)
 (:domain barman)
 (:objects 
      shaker383 - shaker
      left right - hand
      shot240 shot262 - shot
      ingredient352 ingredient272 - ingredient
      cocktail1 - cocktail
      dispenser81 dispenser178 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker383)
  (ontable shot240)
  (ontable shot262)
  (dispenses dispenser81 ingredient352)
  (dispenses dispenser178 ingredient272)
  (clean shaker383)
  (clean shot240)
  (clean shot262)
  (empty shaker383)
  (empty shot240)
  (empty shot262)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker383 l0)
  (shaker-level shaker383 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient352)
  (cocktail-part2 cocktail1 ingredient272)
)
 (:goal
  (and
      (contains shot240 cocktail1)
)))
