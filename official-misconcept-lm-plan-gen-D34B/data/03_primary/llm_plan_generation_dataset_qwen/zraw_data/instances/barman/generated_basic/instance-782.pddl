(define (problem prob)
 (:domain barman)
 (:objects 
      shaker141 - shaker
      left right - hand
      shot180 - shot
      ingredient252 ingredient388 - ingredient
      cocktail1 - cocktail
      dispenser429 dispenser31 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker141)
  (ontable shot180)
  (dispenses dispenser429 ingredient252)
  (dispenses dispenser31 ingredient388)
  (clean shaker141)
  (clean shot180)
  (empty shaker141)
  (empty shot180)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker141 l0)
  (shaker-level shaker141 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient252)
  (cocktail-part2 cocktail1 ingredient388)
)
 (:goal
  (and
      (contains shot180 cocktail1)
)))
