(define (problem prob)
 (:domain barman)
 (:objects 
      shaker189 - shaker
      left right - hand
      shot481 shot335 - shot
      ingredient197 ingredient389 ingredient41 - ingredient
      cocktail311 - cocktail
      dispenser112 dispenser128 dispenser58 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker189)
  (ontable shot481)
  (ontable shot335)
  (dispenses dispenser112 ingredient197)
  (dispenses dispenser128 ingredient389)
  (dispenses dispenser58 ingredient41)
  (clean shaker189)
  (clean shot481)
  (clean shot335)
  (empty shaker189)
  (empty shot481)
  (empty shot335)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker189 l0)
  (shaker-level shaker189 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail311 ingredient389)
  (cocktail-part2 cocktail311 ingredient41)
)
 (:goal
  (and
      (contains shot481 cocktail311)
)))
