(define (problem prob)
 (:domain barman)
 (:objects 
      shaker370 - shaker
      left right - hand
      shot339 shot169 - shot
      ingredient84 ingredient37 ingredient190 - ingredient
      cocktail465 - cocktail
      dispenser417 dispenser428 dispenser20 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker370)
  (ontable shot339)
  (ontable shot169)
  (dispenses dispenser417 ingredient84)
  (dispenses dispenser428 ingredient37)
  (dispenses dispenser20 ingredient190)
  (clean shaker370)
  (clean shot339)
  (clean shot169)
  (empty shaker370)
  (empty shot339)
  (empty shot169)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker370 l0)
  (shaker-level shaker370 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail465 ingredient190)
  (cocktail-part2 cocktail465 ingredient37)
)
 (:goal
  (and
      (contains shot339 cocktail465)
)))
