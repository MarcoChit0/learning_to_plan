(define (problem prob)
 (:domain barman)
 (:objects 
      shaker63 - shaker
      left right - hand
      shot155 shot156 shot193 - shot
      ingredient363 ingredient214 - ingredient
      cocktail292 - cocktail
      dispenser127 dispenser348 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker63)
  (ontable shot155)
  (ontable shot156)
  (ontable shot193)
  (dispenses dispenser127 ingredient363)
  (dispenses dispenser348 ingredient214)
  (clean shaker63)
  (clean shot155)
  (clean shot156)
  (clean shot193)
  (empty shaker63)
  (empty shot155)
  (empty shot156)
  (empty shot193)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker63 l0)
  (shaker-level shaker63 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail292 ingredient363)
  (cocktail-part2 cocktail292 ingredient214)
)
 (:goal
  (and
      (contains shot155 cocktail292)
      (contains shot156 cocktail292)
)))
