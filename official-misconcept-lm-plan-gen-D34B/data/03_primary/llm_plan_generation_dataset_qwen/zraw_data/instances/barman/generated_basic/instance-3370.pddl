(define (problem prob)
 (:domain barman)
 (:objects 
      shaker89 - shaker
      left right - hand
      shot409 shot4 shot375 shot47 - shot
      ingredient430 ingredient93 ingredient342 ingredient410 - ingredient
      cocktail134 - cocktail
      dispenser39 dispenser23 dispenser458 dispenser267 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker89)
  (ontable shot409)
  (ontable shot4)
  (ontable shot375)
  (ontable shot47)
  (dispenses dispenser39 ingredient430)
  (dispenses dispenser23 ingredient93)
  (dispenses dispenser458 ingredient342)
  (dispenses dispenser267 ingredient410)
  (clean shaker89)
  (clean shot409)
  (clean shot4)
  (clean shot375)
  (clean shot47)
  (empty shaker89)
  (empty shot409)
  (empty shot4)
  (empty shot375)
  (empty shot47)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker89 l0)
  (shaker-level shaker89 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail134 ingredient342)
  (cocktail-part2 cocktail134 ingredient410)
)
 (:goal
  (and
      (contains shot409 cocktail134)
      (contains shot4 cocktail134)
      (contains shot375 ingredient410)
)))
