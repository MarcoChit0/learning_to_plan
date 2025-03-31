(define (problem prob)
 (:domain barman)
 (:objects 
      shaker365 - shaker
      left right - hand
      shot129 shot77 shot52 - shot
      ingredient39 ingredient316 ingredient322 ingredient481 - ingredient
      cocktail1 - cocktail
      dispenser309 dispenser169 dispenser151 dispenser116 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker365)
  (ontable shot129)
  (ontable shot77)
  (ontable shot52)
  (dispenses dispenser309 ingredient39)
  (dispenses dispenser169 ingredient316)
  (dispenses dispenser151 ingredient322)
  (dispenses dispenser116 ingredient481)
  (clean shaker365)
  (clean shot129)
  (clean shot77)
  (clean shot52)
  (empty shaker365)
  (empty shot129)
  (empty shot77)
  (empty shot52)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker365 l0)
  (shaker-level shaker365 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient481)
  (cocktail-part2 cocktail1 ingredient39)
)
 (:goal
  (and
      (contains shot129 cocktail1)
      (contains shot77 cocktail1)
)))
