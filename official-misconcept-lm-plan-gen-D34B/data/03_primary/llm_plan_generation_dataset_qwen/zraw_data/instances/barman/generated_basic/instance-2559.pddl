(define (problem prob)
 (:domain barman)
 (:objects 
      shaker330 - shaker
      left right - hand
      shot51 shot167 shot294 - shot
      ingredient47 ingredient375 - ingredient
      cocktail475 - cocktail
      dispenser276 dispenser25 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker330)
  (ontable shot51)
  (ontable shot167)
  (ontable shot294)
  (dispenses dispenser276 ingredient47)
  (dispenses dispenser25 ingredient375)
  (clean shaker330)
  (clean shot51)
  (clean shot167)
  (clean shot294)
  (empty shaker330)
  (empty shot51)
  (empty shot167)
  (empty shot294)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker330 l0)
  (shaker-level shaker330 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail475 ingredient47)
  (cocktail-part2 cocktail475 ingredient375)
)
 (:goal
  (and
      (contains shot51 cocktail475)
      (contains shot167 cocktail475)
)))
