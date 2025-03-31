(define (problem prob)
 (:domain barman)
 (:objects 
      shaker126 - shaker
      left right - hand
      shot397 - shot
      ingredient163 ingredient63 ingredient228 - ingredient
      cocktail150 - cocktail
      dispenser38 dispenser194 dispenser122 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker126)
  (ontable shot397)
  (dispenses dispenser38 ingredient163)
  (dispenses dispenser194 ingredient63)
  (dispenses dispenser122 ingredient228)
  (clean shaker126)
  (clean shot397)
  (empty shaker126)
  (empty shot397)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker126 l0)
  (shaker-level shaker126 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail150 ingredient63)
  (cocktail-part2 cocktail150 ingredient228)
)
 (:goal
  (and
      (contains shot397 cocktail150)
)))
