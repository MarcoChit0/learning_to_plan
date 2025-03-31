(define (problem prob)
 (:domain barman)
 (:objects 
      shaker392 - shaker
      left right - hand
      shot297 shot101 - shot
      ingredient146 ingredient85 ingredient266 ingredient81 - ingredient
      cocktail352 - cocktail
      dispenser388 dispenser13 dispenser299 dispenser209 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker392)
  (ontable shot297)
  (ontable shot101)
  (dispenses dispenser388 ingredient146)
  (dispenses dispenser13 ingredient85)
  (dispenses dispenser299 ingredient266)
  (dispenses dispenser209 ingredient81)
  (clean shaker392)
  (clean shot297)
  (clean shot101)
  (empty shaker392)
  (empty shot297)
  (empty shot101)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker392 l0)
  (shaker-level shaker392 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail352 ingredient266)
  (cocktail-part2 cocktail352 ingredient85)
)
 (:goal
  (and
      (contains shot297 cocktail352)
)))
