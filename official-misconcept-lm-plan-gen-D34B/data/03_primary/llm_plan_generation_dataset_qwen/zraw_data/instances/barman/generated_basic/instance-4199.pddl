(define (problem prob)
 (:domain barman)
 (:objects 
      shaker320 - shaker
      left right - hand
      shot180 shot292 shot267 - shot
      ingredient247 ingredient381 ingredient432 - ingredient
      cocktail175 - cocktail
      dispenser183 dispenser76 dispenser420 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker320)
  (ontable shot180)
  (ontable shot292)
  (ontable shot267)
  (dispenses dispenser183 ingredient247)
  (dispenses dispenser76 ingredient381)
  (dispenses dispenser420 ingredient432)
  (clean shaker320)
  (clean shot180)
  (clean shot292)
  (clean shot267)
  (empty shaker320)
  (empty shot180)
  (empty shot292)
  (empty shot267)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker320 l0)
  (shaker-level shaker320 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail175 ingredient432)
  (cocktail-part2 cocktail175 ingredient381)
)
 (:goal
  (and
      (contains shot180 cocktail175)
      (contains shot292 cocktail175)
)))
