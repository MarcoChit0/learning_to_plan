(define (problem prob)
 (:domain barman)
 (:objects 
      shaker380 - shaker
      left right - hand
      shot467 - shot
      ingredient292 ingredient118 ingredient82 - ingredient
      cocktail1 - cocktail
      dispenser0 dispenser455 dispenser277 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker380)
  (ontable shot467)
  (dispenses dispenser0 ingredient292)
  (dispenses dispenser455 ingredient118)
  (dispenses dispenser277 ingredient82)
  (clean shaker380)
  (clean shot467)
  (empty shaker380)
  (empty shot467)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker380 l0)
  (shaker-level shaker380 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient118)
  (cocktail-part2 cocktail1 ingredient82)
)
 (:goal
  (and
      (contains shot467 cocktail1)
)))
