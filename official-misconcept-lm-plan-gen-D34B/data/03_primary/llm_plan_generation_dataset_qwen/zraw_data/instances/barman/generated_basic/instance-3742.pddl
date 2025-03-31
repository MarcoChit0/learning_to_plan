(define (problem prob)
 (:domain barman)
 (:objects 
      shaker438 - shaker
      left right - hand
      shot231 shot469 shot80 - shot
      ingredient176 ingredient388 ingredient123 - ingredient
      cocktail204 - cocktail
      dispenser233 dispenser174 dispenser270 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker438)
  (ontable shot231)
  (ontable shot469)
  (ontable shot80)
  (dispenses dispenser233 ingredient176)
  (dispenses dispenser174 ingredient388)
  (dispenses dispenser270 ingredient123)
  (clean shaker438)
  (clean shot231)
  (clean shot469)
  (clean shot80)
  (empty shaker438)
  (empty shot231)
  (empty shot469)
  (empty shot80)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker438 l0)
  (shaker-level shaker438 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail204 ingredient123)
  (cocktail-part2 cocktail204 ingredient176)
)
 (:goal
  (and
      (contains shot231 cocktail204)
      (contains shot469 ingredient176)
)))
