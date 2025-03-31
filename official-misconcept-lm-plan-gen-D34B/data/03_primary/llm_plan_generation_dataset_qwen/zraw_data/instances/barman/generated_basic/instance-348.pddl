(define (problem prob)
 (:domain barman)
 (:objects 
      shaker156 - shaker
      left right - hand
      shot451 shot361 - shot
      ingredient215 ingredient229 ingredient449 ingredient91 - ingredient
      cocktail1 - cocktail
      dispenser264 dispenser495 dispenser319 dispenser303 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker156)
  (ontable shot451)
  (ontable shot361)
  (dispenses dispenser264 ingredient215)
  (dispenses dispenser495 ingredient229)
  (dispenses dispenser319 ingredient449)
  (dispenses dispenser303 ingredient91)
  (clean shaker156)
  (clean shot451)
  (clean shot361)
  (empty shaker156)
  (empty shot451)
  (empty shot361)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker156 l0)
  (shaker-level shaker156 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient215)
  (cocktail-part2 cocktail1 ingredient449)
)
 (:goal
  (and
      (contains shot451 cocktail1)
)))
