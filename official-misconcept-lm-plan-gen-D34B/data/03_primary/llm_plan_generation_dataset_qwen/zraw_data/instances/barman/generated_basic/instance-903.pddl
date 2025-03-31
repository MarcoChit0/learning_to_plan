(define (problem prob)
 (:domain barman)
 (:objects 
      shaker212 - shaker
      left right - hand
      shot384 shot210 - shot
      ingredient293 ingredient34 - ingredient
      cocktail1 - cocktail
      dispenser73 dispenser52 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker212)
  (ontable shot384)
  (ontable shot210)
  (dispenses dispenser73 ingredient293)
  (dispenses dispenser52 ingredient34)
  (clean shaker212)
  (clean shot384)
  (clean shot210)
  (empty shaker212)
  (empty shot384)
  (empty shot210)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker212 l0)
  (shaker-level shaker212 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient34)
  (cocktail-part2 cocktail1 ingredient293)
)
 (:goal
  (and
      (contains shot384 cocktail1)
)))
