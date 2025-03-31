(define (problem prob)
 (:domain barman)
 (:objects 
      shaker146 - shaker
      left right - hand
      shot409 - shot
      ingredient170 ingredient441 - ingredient
      cocktail190 - cocktail
      dispenser188 dispenser123 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker146)
  (ontable shot409)
  (dispenses dispenser188 ingredient170)
  (dispenses dispenser123 ingredient441)
  (clean shaker146)
  (clean shot409)
  (empty shaker146)
  (empty shot409)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker146 l0)
  (shaker-level shaker146 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail190 ingredient441)
  (cocktail-part2 cocktail190 ingredient170)
)
 (:goal
  (and
      (contains shot409 cocktail190)
)))
