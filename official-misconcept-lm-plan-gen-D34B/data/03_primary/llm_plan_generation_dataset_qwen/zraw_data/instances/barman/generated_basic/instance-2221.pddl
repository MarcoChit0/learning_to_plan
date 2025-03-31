(define (problem prob)
 (:domain barman)
 (:objects 
      shaker166 - shaker
      left right - hand
      shot89 - shot
      ingredient206 ingredient100 - ingredient
      cocktail440 - cocktail
      dispenser231 dispenser154 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker166)
  (ontable shot89)
  (dispenses dispenser231 ingredient206)
  (dispenses dispenser154 ingredient100)
  (clean shaker166)
  (clean shot89)
  (empty shaker166)
  (empty shot89)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker166 l0)
  (shaker-level shaker166 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail440 ingredient206)
  (cocktail-part2 cocktail440 ingredient100)
)
 (:goal
  (and
      (contains shot89 cocktail440)
)))
