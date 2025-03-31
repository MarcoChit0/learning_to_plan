(define (problem prob)
 (:domain barman)
 (:objects 
      shaker235 - shaker
      left right - hand
      shot77 shot129 - shot
      ingredient318 ingredient17 ingredient363 - ingredient
      cocktail1 - cocktail
      dispenser411 dispenser425 dispenser406 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker235)
  (ontable shot77)
  (ontable shot129)
  (dispenses dispenser411 ingredient318)
  (dispenses dispenser425 ingredient17)
  (dispenses dispenser406 ingredient363)
  (clean shaker235)
  (clean shot77)
  (clean shot129)
  (empty shaker235)
  (empty shot77)
  (empty shot129)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker235 l0)
  (shaker-level shaker235 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient318)
  (cocktail-part2 cocktail1 ingredient17)
)
 (:goal
  (and
      (contains shot77 cocktail1)
)))
