(define (problem prob)
 (:domain barman)
 (:objects 
      shaker393 - shaker
      left right - hand
      shot62 shot458 shot207 - shot
      ingredient151 ingredient309 - ingredient
      cocktail1 cocktail2 - cocktail
      dispenser30 dispenser52 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker393)
  (ontable shot62)
  (ontable shot458)
  (ontable shot207)
  (dispenses dispenser30 ingredient151)
  (dispenses dispenser52 ingredient309)
  (clean shaker393)
  (clean shot62)
  (clean shot458)
  (clean shot207)
  (empty shaker393)
  (empty shot62)
  (empty shot458)
  (empty shot207)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker393 l0)
  (shaker-level shaker393 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient309)
  (cocktail-part2 cocktail1 ingredient151)
  (cocktail-part1 cocktail2 ingredient309)
  (cocktail-part2 cocktail2 ingredient151)
)
 (:goal
  (and
      (contains shot62 cocktail1)
      (contains shot458 cocktail2)
)))
