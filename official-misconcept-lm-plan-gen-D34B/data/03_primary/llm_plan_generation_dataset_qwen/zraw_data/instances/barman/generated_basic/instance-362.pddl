(define (problem prob)
 (:domain barman)
 (:objects 
      shaker90 - shaker
      left right - hand
      shot294 - shot
      ingredient151 ingredient38 - ingredient
      cocktail1 - cocktail
      dispenser421 dispenser147 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker90)
  (ontable shot294)
  (dispenses dispenser421 ingredient151)
  (dispenses dispenser147 ingredient38)
  (clean shaker90)
  (clean shot294)
  (empty shaker90)
  (empty shot294)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker90 l0)
  (shaker-level shaker90 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient151)
  (cocktail-part2 cocktail1 ingredient38)
)
 (:goal
  (and
      (contains shot294 cocktail1)
)))
