(define (problem prob)
 (:domain barman)
 (:objects 
      shaker432 - shaker
      left right - hand
      shot405 shot269 - shot
      ingredient469 ingredient22 - ingredient
      cocktail436 - cocktail
      dispenser206 dispenser464 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker432)
  (ontable shot405)
  (ontable shot269)
  (dispenses dispenser206 ingredient469)
  (dispenses dispenser464 ingredient22)
  (clean shaker432)
  (clean shot405)
  (clean shot269)
  (empty shaker432)
  (empty shot405)
  (empty shot269)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker432 l0)
  (shaker-level shaker432 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail436 ingredient469)
  (cocktail-part2 cocktail436 ingredient22)
)
 (:goal
  (and
      (contains shot405 cocktail436)
)))
