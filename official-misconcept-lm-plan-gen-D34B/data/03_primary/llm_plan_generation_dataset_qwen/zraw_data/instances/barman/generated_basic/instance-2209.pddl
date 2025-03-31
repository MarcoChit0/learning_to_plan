(define (problem prob)
 (:domain barman)
 (:objects 
      shaker188 - shaker
      left right - hand
      shot422 shot113 shot294 - shot
      ingredient53 ingredient258 ingredient470 ingredient150 - ingredient
      cocktail261 - cocktail
      dispenser80 dispenser469 dispenser37 dispenser415 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker188)
  (ontable shot422)
  (ontable shot113)
  (ontable shot294)
  (dispenses dispenser80 ingredient53)
  (dispenses dispenser469 ingredient258)
  (dispenses dispenser37 ingredient470)
  (dispenses dispenser415 ingredient150)
  (clean shaker188)
  (clean shot422)
  (clean shot113)
  (clean shot294)
  (empty shaker188)
  (empty shot422)
  (empty shot113)
  (empty shot294)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker188 l0)
  (shaker-level shaker188 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail261 ingredient150)
  (cocktail-part2 cocktail261 ingredient53)
)
 (:goal
  (and
      (contains shot422 cocktail261)
      (contains shot113 ingredient150)
)))
