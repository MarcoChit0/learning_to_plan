(define (problem prob)
 (:domain barman)
 (:objects 
      shaker112 - shaker
      left right - hand
      shot335 shot137 shot231 - shot
      ingredient343 ingredient341 ingredient201 ingredient159 - ingredient
      cocktail13 - cocktail
      dispenser221 dispenser492 dispenser424 dispenser337 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker112)
  (ontable shot335)
  (ontable shot137)
  (ontable shot231)
  (dispenses dispenser221 ingredient343)
  (dispenses dispenser492 ingredient341)
  (dispenses dispenser424 ingredient201)
  (dispenses dispenser337 ingredient159)
  (clean shaker112)
  (clean shot335)
  (clean shot137)
  (clean shot231)
  (empty shaker112)
  (empty shot335)
  (empty shot137)
  (empty shot231)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker112 l0)
  (shaker-level shaker112 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail13 ingredient343)
  (cocktail-part2 cocktail13 ingredient201)
)
 (:goal
  (and
      (contains shot335 cocktail13)
      (contains shot137 cocktail13)
)))
