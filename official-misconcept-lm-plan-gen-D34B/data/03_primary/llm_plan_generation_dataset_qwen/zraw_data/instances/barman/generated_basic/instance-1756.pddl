(define (problem prob)
 (:domain barman)
 (:objects 
      shaker416 - shaker
      left right - hand
      shot91 shot326 shot46 - shot
      ingredient99 ingredient253 - ingredient
      cocktail378 - cocktail
      dispenser350 dispenser260 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker416)
  (ontable shot91)
  (ontable shot326)
  (ontable shot46)
  (dispenses dispenser350 ingredient99)
  (dispenses dispenser260 ingredient253)
  (clean shaker416)
  (clean shot91)
  (clean shot326)
  (clean shot46)
  (empty shaker416)
  (empty shot91)
  (empty shot326)
  (empty shot46)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker416 l0)
  (shaker-level shaker416 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail378 ingredient253)
  (cocktail-part2 cocktail378 ingredient99)
)
 (:goal
  (and
      (contains shot91 cocktail378)
      (contains shot326 ingredient253)
)))
