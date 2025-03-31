(define (problem prob)
 (:domain barman)
 (:objects 
      shaker106 - shaker
      left right - hand
      shot472 shot360 shot10 - shot
      ingredient35 ingredient242 ingredient81 - ingredient
      cocktail355 - cocktail
      dispenser234 dispenser30 dispenser474 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker106)
  (ontable shot472)
  (ontable shot360)
  (ontable shot10)
  (dispenses dispenser234 ingredient35)
  (dispenses dispenser30 ingredient242)
  (dispenses dispenser474 ingredient81)
  (clean shaker106)
  (clean shot472)
  (clean shot360)
  (clean shot10)
  (empty shaker106)
  (empty shot472)
  (empty shot360)
  (empty shot10)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker106 l0)
  (shaker-level shaker106 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail355 ingredient81)
  (cocktail-part2 cocktail355 ingredient242)
)
 (:goal
  (and
      (contains shot472 cocktail355)
      (contains shot360 ingredient35)
)))
