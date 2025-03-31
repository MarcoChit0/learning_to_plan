(define (problem prob)
 (:domain barman)
 (:objects 
      shaker15 - shaker
      left right - hand
      shot319 - shot
      ingredient214 ingredient7 ingredient82 - ingredient
      cocktail293 - cocktail
      dispenser171 dispenser354 dispenser24 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker15)
  (ontable shot319)
  (dispenses dispenser171 ingredient214)
  (dispenses dispenser354 ingredient7)
  (dispenses dispenser24 ingredient82)
  (clean shaker15)
  (clean shot319)
  (empty shaker15)
  (empty shot319)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker15 l0)
  (shaker-level shaker15 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail293 ingredient82)
  (cocktail-part2 cocktail293 ingredient214)
)
 (:goal
  (and
      (contains shot319 cocktail293)
)))
