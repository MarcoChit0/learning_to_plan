(define (problem prob)
 (:domain barman)
 (:objects 
      shaker475 - shaker
      left right - hand
      shot267 - shot
      ingredient83 ingredient272 ingredient263 ingredient423 - ingredient
      cocktail114 - cocktail
      dispenser364 dispenser405 dispenser168 dispenser176 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker475)
  (ontable shot267)
  (dispenses dispenser364 ingredient83)
  (dispenses dispenser405 ingredient272)
  (dispenses dispenser168 ingredient263)
  (dispenses dispenser176 ingredient423)
  (clean shaker475)
  (clean shot267)
  (empty shaker475)
  (empty shot267)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker475 l0)
  (shaker-level shaker475 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail114 ingredient272)
  (cocktail-part2 cocktail114 ingredient423)
)
 (:goal
  (and
      (contains shot267 cocktail114)
)))
