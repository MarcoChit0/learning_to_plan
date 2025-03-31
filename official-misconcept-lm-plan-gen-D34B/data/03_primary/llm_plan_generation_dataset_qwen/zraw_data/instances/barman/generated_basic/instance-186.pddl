(define (problem prob)
 (:domain barman)
 (:objects 
      shaker287 - shaker
      left right - hand
      shot253 shot472 shot448 - shot
      ingredient316 ingredient324 ingredient334 ingredient351 - ingredient
      cocktail1 - cocktail
      dispenser223 dispenser329 dispenser81 dispenser446 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker287)
  (ontable shot253)
  (ontable shot472)
  (ontable shot448)
  (dispenses dispenser223 ingredient316)
  (dispenses dispenser329 ingredient324)
  (dispenses dispenser81 ingredient334)
  (dispenses dispenser446 ingredient351)
  (clean shaker287)
  (clean shot253)
  (clean shot472)
  (clean shot448)
  (empty shaker287)
  (empty shot253)
  (empty shot472)
  (empty shot448)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker287 l0)
  (shaker-level shaker287 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient351)
  (cocktail-part2 cocktail1 ingredient324)
)
 (:goal
  (and
      (contains shot253 cocktail1)
      (contains shot472 ingredient316)
)))
