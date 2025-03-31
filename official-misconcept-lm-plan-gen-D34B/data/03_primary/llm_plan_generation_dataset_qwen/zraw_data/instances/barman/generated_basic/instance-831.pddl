(define (problem prob)
 (:domain barman)
 (:objects 
      shaker325 - shaker
      left right - hand
      shot233 shot449 shot334 - shot
      ingredient228 ingredient132 ingredient213 ingredient88 - ingredient
      cocktail1 - cocktail
      dispenser475 dispenser219 dispenser493 dispenser288 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker325)
  (ontable shot233)
  (ontable shot449)
  (ontable shot334)
  (dispenses dispenser475 ingredient228)
  (dispenses dispenser219 ingredient132)
  (dispenses dispenser493 ingredient213)
  (dispenses dispenser288 ingredient88)
  (clean shaker325)
  (clean shot233)
  (clean shot449)
  (clean shot334)
  (empty shaker325)
  (empty shot233)
  (empty shot449)
  (empty shot334)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker325 l0)
  (shaker-level shaker325 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient213)
  (cocktail-part2 cocktail1 ingredient228)
)
 (:goal
  (and
      (contains shot233 cocktail1)
      (contains shot449 ingredient132)
)))
