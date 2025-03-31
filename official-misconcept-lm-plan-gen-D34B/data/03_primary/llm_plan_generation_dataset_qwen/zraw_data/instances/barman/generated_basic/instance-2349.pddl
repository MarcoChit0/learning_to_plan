(define (problem prob)
 (:domain barman)
 (:objects 
      shaker427 - shaker
      left right - hand
      shot187 shot64 - shot
      ingredient484 ingredient96 ingredient288 ingredient342 - ingredient
      cocktail165 - cocktail
      dispenser338 dispenser116 dispenser250 dispenser271 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker427)
  (ontable shot187)
  (ontable shot64)
  (dispenses dispenser338 ingredient484)
  (dispenses dispenser116 ingredient96)
  (dispenses dispenser250 ingredient288)
  (dispenses dispenser271 ingredient342)
  (clean shaker427)
  (clean shot187)
  (clean shot64)
  (empty shaker427)
  (empty shot187)
  (empty shot64)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker427 l0)
  (shaker-level shaker427 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail165 ingredient288)
  (cocktail-part2 cocktail165 ingredient96)
)
 (:goal
  (and
      (contains shot187 cocktail165)
)))
