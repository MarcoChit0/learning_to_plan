(define (problem prob)
 (:domain barman)
 (:objects 
      shaker464 - shaker
      left right - hand
      shot425 shot285 - shot
      ingredient197 ingredient285 - ingredient
      cocktail1 - cocktail
      dispenser64 dispenser414 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker464)
  (ontable shot425)
  (ontable shot285)
  (dispenses dispenser64 ingredient197)
  (dispenses dispenser414 ingredient285)
  (clean shaker464)
  (clean shot425)
  (clean shot285)
  (empty shaker464)
  (empty shot425)
  (empty shot285)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker464 l0)
  (shaker-level shaker464 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient285)
  (cocktail-part2 cocktail1 ingredient197)
)
 (:goal
  (and
      (contains shot425 cocktail1)
)))
