(define (problem prob)
 (:domain barman)
 (:objects 
      shaker481 - shaker
      left right - hand
      shot464 shot216 shot89 - shot
      ingredient76 ingredient405 - ingredient
      cocktail265 - cocktail
      dispenser340 dispenser161 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker481)
  (ontable shot464)
  (ontable shot216)
  (ontable shot89)
  (dispenses dispenser340 ingredient76)
  (dispenses dispenser161 ingredient405)
  (clean shaker481)
  (clean shot464)
  (clean shot216)
  (clean shot89)
  (empty shaker481)
  (empty shot464)
  (empty shot216)
  (empty shot89)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker481 l0)
  (shaker-level shaker481 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail265 ingredient76)
  (cocktail-part2 cocktail265 ingredient405)
)
 (:goal
  (and
      (contains shot464 cocktail265)
      (contains shot216 ingredient76)
)))
