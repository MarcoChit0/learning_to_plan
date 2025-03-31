(define (problem prob)
 (:domain barman)
 (:objects 
      shaker360 - shaker
      left right - hand
      shot403 - shot
      ingredient342 ingredient30 ingredient355 ingredient32 - ingredient
      cocktail90 - cocktail
      dispenser33 dispenser114 dispenser67 dispenser110 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker360)
  (ontable shot403)
  (dispenses dispenser33 ingredient342)
  (dispenses dispenser114 ingredient30)
  (dispenses dispenser67 ingredient355)
  (dispenses dispenser110 ingredient32)
  (clean shaker360)
  (clean shot403)
  (empty shaker360)
  (empty shot403)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker360 l0)
  (shaker-level shaker360 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail90 ingredient342)
  (cocktail-part2 cocktail90 ingredient30)
)
 (:goal
  (and
      (contains shot403 cocktail90)
)))
