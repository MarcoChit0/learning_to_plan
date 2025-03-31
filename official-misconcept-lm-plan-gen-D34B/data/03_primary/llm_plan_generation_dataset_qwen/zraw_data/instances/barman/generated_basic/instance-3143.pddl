(define (problem prob)
 (:domain barman)
 (:objects 
      shaker398 - shaker
      left right - hand
      shot180 shot358 shot423 - shot
      ingredient37 ingredient487 - ingredient
      cocktail328 - cocktail
      dispenser291 dispenser245 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker398)
  (ontable shot180)
  (ontable shot358)
  (ontable shot423)
  (dispenses dispenser291 ingredient37)
  (dispenses dispenser245 ingredient487)
  (clean shaker398)
  (clean shot180)
  (clean shot358)
  (clean shot423)
  (empty shaker398)
  (empty shot180)
  (empty shot358)
  (empty shot423)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker398 l0)
  (shaker-level shaker398 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail328 ingredient37)
  (cocktail-part2 cocktail328 ingredient487)
)
 (:goal
  (and
      (contains shot180 cocktail328)
      (contains shot358 ingredient487)
)))
