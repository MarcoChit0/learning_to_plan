(define (problem prob)
 (:domain barman)
 (:objects 
      shaker109 - shaker
      left right - hand
      shot243 - shot
      ingredient326 ingredient190 ingredient380 ingredient163 - ingredient
      cocktail388 - cocktail
      dispenser494 dispenser289 dispenser120 dispenser426 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker109)
  (ontable shot243)
  (dispenses dispenser494 ingredient326)
  (dispenses dispenser289 ingredient190)
  (dispenses dispenser120 ingredient380)
  (dispenses dispenser426 ingredient163)
  (clean shaker109)
  (clean shot243)
  (empty shaker109)
  (empty shot243)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker109 l0)
  (shaker-level shaker109 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail388 ingredient326)
  (cocktail-part2 cocktail388 ingredient163)
)
 (:goal
  (and
      (contains shot243 cocktail388)
)))
