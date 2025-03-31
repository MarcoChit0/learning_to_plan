(define (problem prob)
 (:domain barman)
 (:objects 
      shaker375 - shaker
      left right - hand
      shot316 shot54 - shot
      ingredient331 ingredient47 ingredient82 - ingredient
      cocktail279 - cocktail
      dispenser335 dispenser192 dispenser30 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker375)
  (ontable shot316)
  (ontable shot54)
  (dispenses dispenser335 ingredient331)
  (dispenses dispenser192 ingredient47)
  (dispenses dispenser30 ingredient82)
  (clean shaker375)
  (clean shot316)
  (clean shot54)
  (empty shaker375)
  (empty shot316)
  (empty shot54)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker375 l0)
  (shaker-level shaker375 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail279 ingredient331)
  (cocktail-part2 cocktail279 ingredient47)
)
 (:goal
  (and
      (contains shot316 cocktail279)
)))
