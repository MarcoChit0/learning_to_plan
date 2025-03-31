(define (problem prob)
 (:domain barman)
 (:objects 
      shaker66 - shaker
      left right - hand
      shot339 - shot
      ingredient163 ingredient83 ingredient349 ingredient165 - ingredient
      cocktail1 - cocktail
      dispenser427 dispenser230 dispenser464 dispenser56 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker66)
  (ontable shot339)
  (dispenses dispenser427 ingredient163)
  (dispenses dispenser230 ingredient83)
  (dispenses dispenser464 ingredient349)
  (dispenses dispenser56 ingredient165)
  (clean shaker66)
  (clean shot339)
  (empty shaker66)
  (empty shot339)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker66 l0)
  (shaker-level shaker66 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient165)
  (cocktail-part2 cocktail1 ingredient83)
)
 (:goal
  (and
      (contains shot339 cocktail1)
)))
