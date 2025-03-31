(define (problem prob)
 (:domain barman)
 (:objects 
      shaker438 - shaker
      left right - hand
      shot103 - shot
      ingredient158 ingredient4 - ingredient
      cocktail77 - cocktail
      dispenser10 dispenser403 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker438)
  (ontable shot103)
  (dispenses dispenser10 ingredient158)
  (dispenses dispenser403 ingredient4)
  (clean shaker438)
  (clean shot103)
  (empty shaker438)
  (empty shot103)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker438 l0)
  (shaker-level shaker438 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail77 ingredient158)
  (cocktail-part2 cocktail77 ingredient4)
)
 (:goal
  (and
      (contains shot103 cocktail77)
)))
