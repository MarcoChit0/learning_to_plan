(define (problem prob)
 (:domain barman)
 (:objects 
      shaker113 - shaker
      left right - hand
      shot449 shot362 - shot
      ingredient219 ingredient225 ingredient182 ingredient4 - ingredient
      cocktail473 - cocktail
      dispenser99 dispenser316 dispenser28 dispenser266 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker113)
  (ontable shot449)
  (ontable shot362)
  (dispenses dispenser99 ingredient219)
  (dispenses dispenser316 ingredient225)
  (dispenses dispenser28 ingredient182)
  (dispenses dispenser266 ingredient4)
  (clean shaker113)
  (clean shot449)
  (clean shot362)
  (empty shaker113)
  (empty shot449)
  (empty shot362)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker113 l0)
  (shaker-level shaker113 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail473 ingredient4)
  (cocktail-part2 cocktail473 ingredient225)
)
 (:goal
  (and
      (contains shot449 cocktail473)
)))
