(define (problem prob)
 (:domain barman)
 (:objects 
      shaker237 - shaker
      left right - hand
      shot454 shot429 - shot
      ingredient458 ingredient451 - ingredient
      cocktail446 - cocktail
      dispenser387 dispenser293 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker237)
  (ontable shot454)
  (ontable shot429)
  (dispenses dispenser387 ingredient458)
  (dispenses dispenser293 ingredient451)
  (clean shaker237)
  (clean shot454)
  (clean shot429)
  (empty shaker237)
  (empty shot454)
  (empty shot429)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker237 l0)
  (shaker-level shaker237 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail446 ingredient451)
  (cocktail-part2 cocktail446 ingredient458)
)
 (:goal
  (and
      (contains shot454 cocktail446)
)))
