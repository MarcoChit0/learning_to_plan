(define (problem prob)
 (:domain barman)
 (:objects 
      shaker481 - shaker
      left right - hand
      shot329 shot89 - shot
      ingredient352 ingredient312 ingredient287 ingredient390 - ingredient
      cocktail1 - cocktail
      dispenser192 dispenser89 dispenser367 dispenser121 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker481)
  (ontable shot329)
  (ontable shot89)
  (dispenses dispenser192 ingredient352)
  (dispenses dispenser89 ingredient312)
  (dispenses dispenser367 ingredient287)
  (dispenses dispenser121 ingredient390)
  (clean shaker481)
  (clean shot329)
  (clean shot89)
  (empty shaker481)
  (empty shot329)
  (empty shot89)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker481 l0)
  (shaker-level shaker481 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient352)
  (cocktail-part2 cocktail1 ingredient312)
)
 (:goal
  (and
      (contains shot329 cocktail1)
)))
