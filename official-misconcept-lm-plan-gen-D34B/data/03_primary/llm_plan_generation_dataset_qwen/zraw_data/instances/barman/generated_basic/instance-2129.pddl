(define (problem prob)
 (:domain barman)
 (:objects 
      shaker9 - shaker
      left right - hand
      shot283 shot280 shot130 - shot
      ingredient460 ingredient54 - ingredient
      cocktail289 - cocktail
      dispenser472 dispenser266 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker9)
  (ontable shot283)
  (ontable shot280)
  (ontable shot130)
  (dispenses dispenser472 ingredient460)
  (dispenses dispenser266 ingredient54)
  (clean shaker9)
  (clean shot283)
  (clean shot280)
  (clean shot130)
  (empty shaker9)
  (empty shot283)
  (empty shot280)
  (empty shot130)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker9 l0)
  (shaker-level shaker9 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail289 ingredient460)
  (cocktail-part2 cocktail289 ingredient54)
)
 (:goal
  (and
      (contains shot283 cocktail289)
      (contains shot280 cocktail289)
)))
