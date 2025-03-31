(define (problem prob)
 (:domain barman)
 (:objects 
      shaker228 - shaker
      left right - hand
      shot249 - shot
      ingredient231 ingredient12 ingredient198 ingredient344 - ingredient
      cocktail376 - cocktail
      dispenser404 dispenser9 dispenser17 dispenser374 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker228)
  (ontable shot249)
  (dispenses dispenser404 ingredient231)
  (dispenses dispenser9 ingredient12)
  (dispenses dispenser17 ingredient198)
  (dispenses dispenser374 ingredient344)
  (clean shaker228)
  (clean shot249)
  (empty shaker228)
  (empty shot249)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker228 l0)
  (shaker-level shaker228 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail376 ingredient344)
  (cocktail-part2 cocktail376 ingredient12)
)
 (:goal
  (and
      (contains shot249 cocktail376)
)))
