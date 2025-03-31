(define (problem prob)
 (:domain barman)
 (:objects 
      shaker428 - shaker
      left right - hand
      shot299 shot60 - shot
      ingredient493 ingredient230 - ingredient
      cocktail414 - cocktail
      dispenser305 dispenser472 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker428)
  (ontable shot299)
  (ontable shot60)
  (dispenses dispenser305 ingredient493)
  (dispenses dispenser472 ingredient230)
  (clean shaker428)
  (clean shot299)
  (clean shot60)
  (empty shaker428)
  (empty shot299)
  (empty shot60)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker428 l0)
  (shaker-level shaker428 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail414 ingredient230)
  (cocktail-part2 cocktail414 ingredient493)
)
 (:goal
  (and
      (contains shot299 cocktail414)
)))
