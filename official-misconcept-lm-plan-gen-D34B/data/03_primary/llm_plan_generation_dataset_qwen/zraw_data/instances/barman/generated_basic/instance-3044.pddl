(define (problem prob)
 (:domain barman)
 (:objects 
      shaker125 - shaker
      left right - hand
      shot110 shot64 - shot
      ingredient151 ingredient425 ingredient320 - ingredient
      cocktail214 - cocktail
      dispenser392 dispenser429 dispenser350 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker125)
  (ontable shot110)
  (ontable shot64)
  (dispenses dispenser392 ingredient151)
  (dispenses dispenser429 ingredient425)
  (dispenses dispenser350 ingredient320)
  (clean shaker125)
  (clean shot110)
  (clean shot64)
  (empty shaker125)
  (empty shot110)
  (empty shot64)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker125 l0)
  (shaker-level shaker125 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail214 ingredient151)
  (cocktail-part2 cocktail214 ingredient320)
)
 (:goal
  (and
      (contains shot110 cocktail214)
)))
