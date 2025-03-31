(define (problem prob)
 (:domain barman)
 (:objects 
      shaker220 - shaker
      left right - hand
      shot177 shot479 - shot
      ingredient105 ingredient394 ingredient40 ingredient490 - ingredient
      cocktail69 - cocktail
      dispenser288 dispenser192 dispenser103 dispenser211 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker220)
  (ontable shot177)
  (ontable shot479)
  (dispenses dispenser288 ingredient105)
  (dispenses dispenser192 ingredient394)
  (dispenses dispenser103 ingredient40)
  (dispenses dispenser211 ingredient490)
  (clean shaker220)
  (clean shot177)
  (clean shot479)
  (empty shaker220)
  (empty shot177)
  (empty shot479)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker220 l0)
  (shaker-level shaker220 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail69 ingredient40)
  (cocktail-part2 cocktail69 ingredient394)
)
 (:goal
  (and
      (contains shot177 cocktail69)
)))
