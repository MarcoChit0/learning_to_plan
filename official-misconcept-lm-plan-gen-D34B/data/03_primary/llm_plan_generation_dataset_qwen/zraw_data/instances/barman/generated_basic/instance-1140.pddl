(define (problem prob)
 (:domain barman)
 (:objects 
      shaker375 - shaker
      left right - hand
      shot382 shot279 - shot
      ingredient428 ingredient145 ingredient300 - ingredient
      cocktail1 - cocktail
      dispenser300 dispenser344 dispenser367 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker375)
  (ontable shot382)
  (ontable shot279)
  (dispenses dispenser300 ingredient428)
  (dispenses dispenser344 ingredient145)
  (dispenses dispenser367 ingredient300)
  (clean shaker375)
  (clean shot382)
  (clean shot279)
  (empty shaker375)
  (empty shot382)
  (empty shot279)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker375 l0)
  (shaker-level shaker375 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient145)
  (cocktail-part2 cocktail1 ingredient300)
)
 (:goal
  (and
      (contains shot382 cocktail1)
)))
