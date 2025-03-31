(define (problem prob)
 (:domain barman)
 (:objects 
      shaker384 - shaker
      left right - hand
      shot357 shot249 - shot
      ingredient478 ingredient431 ingredient168 ingredient228 - ingredient
      cocktail1 - cocktail
      dispenser147 dispenser232 dispenser52 dispenser133 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker384)
  (ontable shot357)
  (ontable shot249)
  (dispenses dispenser147 ingredient478)
  (dispenses dispenser232 ingredient431)
  (dispenses dispenser52 ingredient168)
  (dispenses dispenser133 ingredient228)
  (clean shaker384)
  (clean shot357)
  (clean shot249)
  (empty shaker384)
  (empty shot357)
  (empty shot249)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker384 l0)
  (shaker-level shaker384 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient478)
  (cocktail-part2 cocktail1 ingredient168)
)
 (:goal
  (and
      (contains shot357 cocktail1)
)))
