(define (problem prob)
 (:domain barman)
 (:objects 
      shaker438 - shaker
      left right - hand
      shot255 shot305 shot475 - shot
      ingredient470 ingredient362 - ingredient
      cocktail429 - cocktail
      dispenser70 dispenser470 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker438)
  (ontable shot255)
  (ontable shot305)
  (ontable shot475)
  (dispenses dispenser70 ingredient470)
  (dispenses dispenser470 ingredient362)
  (clean shaker438)
  (clean shot255)
  (clean shot305)
  (clean shot475)
  (empty shaker438)
  (empty shot255)
  (empty shot305)
  (empty shot475)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker438 l0)
  (shaker-level shaker438 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail429 ingredient470)
  (cocktail-part2 cocktail429 ingredient362)
)
 (:goal
  (and
      (contains shot255 cocktail429)
      (contains shot305 cocktail429)
)))
