(define (problem prob)
 (:domain barman)
 (:objects 
      shaker189 - shaker
      left right - hand
      shot350 shot23 shot98 - shot
      ingredient342 ingredient495 - ingredient
      cocktail322 - cocktail
      dispenser368 dispenser59 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker189)
  (ontable shot350)
  (ontable shot23)
  (ontable shot98)
  (dispenses dispenser368 ingredient342)
  (dispenses dispenser59 ingredient495)
  (clean shaker189)
  (clean shot350)
  (clean shot23)
  (clean shot98)
  (empty shaker189)
  (empty shot350)
  (empty shot23)
  (empty shot98)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker189 l0)
  (shaker-level shaker189 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail322 ingredient342)
  (cocktail-part2 cocktail322 ingredient495)
)
 (:goal
  (and
      (contains shot350 cocktail322)
      (contains shot23 cocktail322)
)))
