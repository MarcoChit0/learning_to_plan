(define (problem prob)
 (:domain barman)
 (:objects 
      shaker215 - shaker
      left right - hand
      shot269 - shot
      ingredient79 ingredient324 ingredient15 ingredient478 - ingredient
      cocktail73 - cocktail
      dispenser122 dispenser266 dispenser143 dispenser239 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker215)
  (ontable shot269)
  (dispenses dispenser122 ingredient79)
  (dispenses dispenser266 ingredient324)
  (dispenses dispenser143 ingredient15)
  (dispenses dispenser239 ingredient478)
  (clean shaker215)
  (clean shot269)
  (empty shaker215)
  (empty shot269)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker215 l0)
  (shaker-level shaker215 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail73 ingredient324)
  (cocktail-part2 cocktail73 ingredient478)
)
 (:goal
  (and
      (contains shot269 cocktail73)
)))
