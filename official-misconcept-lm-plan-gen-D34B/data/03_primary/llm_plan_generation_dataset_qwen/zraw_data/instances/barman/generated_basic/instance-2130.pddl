(define (problem prob)
 (:domain barman)
 (:objects 
      shaker123 - shaker
      left right - hand
      shot469 shot495 shot158 - shot
      ingredient437 ingredient150 - ingredient
      cocktail110 - cocktail
      dispenser317 dispenser249 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker123)
  (ontable shot469)
  (ontable shot495)
  (ontable shot158)
  (dispenses dispenser317 ingredient437)
  (dispenses dispenser249 ingredient150)
  (clean shaker123)
  (clean shot469)
  (clean shot495)
  (clean shot158)
  (empty shaker123)
  (empty shot469)
  (empty shot495)
  (empty shot158)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker123 l0)
  (shaker-level shaker123 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail110 ingredient150)
  (cocktail-part2 cocktail110 ingredient437)
)
 (:goal
  (and
      (contains shot469 cocktail110)
      (contains shot495 cocktail110)
)))
