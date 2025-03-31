(define (problem prob)
 (:domain barman)
 (:objects 
      shaker413 - shaker
      left right - hand
      shot140 - shot
      ingredient17 ingredient11 - ingredient
      cocktail190 - cocktail
      dispenser427 dispenser101 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker413)
  (ontable shot140)
  (dispenses dispenser427 ingredient17)
  (dispenses dispenser101 ingredient11)
  (clean shaker413)
  (clean shot140)
  (empty shaker413)
  (empty shot140)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker413 l0)
  (shaker-level shaker413 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail190 ingredient17)
  (cocktail-part2 cocktail190 ingredient11)
)
 (:goal
  (and
      (contains shot140 cocktail190)
)))
