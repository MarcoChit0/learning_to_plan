(define (problem prob)
 (:domain barman)
 (:objects 
      shaker426 - shaker
      left right - hand
      shot344 shot403 shot424 - shot
      ingredient40 ingredient184 ingredient174 - ingredient
      cocktail371 - cocktail
      dispenser193 dispenser131 dispenser45 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker426)
  (ontable shot344)
  (ontable shot403)
  (ontable shot424)
  (dispenses dispenser193 ingredient40)
  (dispenses dispenser131 ingredient184)
  (dispenses dispenser45 ingredient174)
  (clean shaker426)
  (clean shot344)
  (clean shot403)
  (clean shot424)
  (empty shaker426)
  (empty shot344)
  (empty shot403)
  (empty shot424)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker426 l0)
  (shaker-level shaker426 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail371 ingredient40)
  (cocktail-part2 cocktail371 ingredient184)
)
 (:goal
  (and
      (contains shot344 cocktail371)
      (contains shot403 cocktail371)
)))
