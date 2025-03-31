(define (problem prob)
 (:domain barman)
 (:objects 
      shaker108 - shaker
      left right - hand
      shot249 shot259 - shot
      ingredient319 ingredient294 - ingredient
      cocktail255 - cocktail
      dispenser328 dispenser85 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker108)
  (ontable shot249)
  (ontable shot259)
  (dispenses dispenser328 ingredient319)
  (dispenses dispenser85 ingredient294)
  (clean shaker108)
  (clean shot249)
  (clean shot259)
  (empty shaker108)
  (empty shot249)
  (empty shot259)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker108 l0)
  (shaker-level shaker108 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail255 ingredient319)
  (cocktail-part2 cocktail255 ingredient294)
)
 (:goal
  (and
      (contains shot249 cocktail255)
)))
