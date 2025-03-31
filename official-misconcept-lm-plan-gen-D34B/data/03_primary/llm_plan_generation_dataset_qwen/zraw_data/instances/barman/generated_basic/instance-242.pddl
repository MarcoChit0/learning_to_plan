(define (problem prob)
 (:domain barman)
 (:objects 
      shaker82 - shaker
      left right - hand
      shot493 - shot
      ingredient455 ingredient22 - ingredient
      cocktail1 - cocktail
      dispenser239 dispenser325 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker82)
  (ontable shot493)
  (dispenses dispenser239 ingredient455)
  (dispenses dispenser325 ingredient22)
  (clean shaker82)
  (clean shot493)
  (empty shaker82)
  (empty shot493)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker82 l0)
  (shaker-level shaker82 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient455)
  (cocktail-part2 cocktail1 ingredient22)
)
 (:goal
  (and
      (contains shot493 cocktail1)
)))
