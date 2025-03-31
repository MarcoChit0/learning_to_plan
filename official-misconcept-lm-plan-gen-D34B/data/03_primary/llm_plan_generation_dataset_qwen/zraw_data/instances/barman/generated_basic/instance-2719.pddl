(define (problem prob)
 (:domain barman)
 (:objects 
      shaker117 - shaker
      left right - hand
      shot75 shot268 shot405 - shot
      ingredient400 ingredient334 ingredient422 - ingredient
      cocktail164 - cocktail
      dispenser306 dispenser74 dispenser235 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker117)
  (ontable shot75)
  (ontable shot268)
  (ontable shot405)
  (dispenses dispenser306 ingredient400)
  (dispenses dispenser74 ingredient334)
  (dispenses dispenser235 ingredient422)
  (clean shaker117)
  (clean shot75)
  (clean shot268)
  (clean shot405)
  (empty shaker117)
  (empty shot75)
  (empty shot268)
  (empty shot405)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker117 l0)
  (shaker-level shaker117 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail164 ingredient334)
  (cocktail-part2 cocktail164 ingredient422)
)
 (:goal
  (and
      (contains shot75 cocktail164)
      (contains shot268 ingredient400)
)))
