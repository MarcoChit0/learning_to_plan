(define (problem prob)
 (:domain barman)
 (:objects 
      shaker95 - shaker
      left right - hand
      shot486 shot69 shot148 - shot
      ingredient140 ingredient145 ingredient3 - ingredient
      cocktail1 cocktail2 - cocktail
      dispenser0 dispenser219 dispenser47 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker95)
  (ontable shot486)
  (ontable shot69)
  (ontable shot148)
  (dispenses dispenser0 ingredient140)
  (dispenses dispenser219 ingredient145)
  (dispenses dispenser47 ingredient3)
  (clean shaker95)
  (clean shot486)
  (clean shot69)
  (clean shot148)
  (empty shaker95)
  (empty shot486)
  (empty shot69)
  (empty shot148)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker95 l0)
  (shaker-level shaker95 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient145)
  (cocktail-part2 cocktail1 ingredient140)
  (cocktail-part1 cocktail2 ingredient3)
  (cocktail-part2 cocktail2 ingredient145)
)
 (:goal
  (and
      (contains shot486 cocktail1)
      (contains shot69 cocktail2)
)))
