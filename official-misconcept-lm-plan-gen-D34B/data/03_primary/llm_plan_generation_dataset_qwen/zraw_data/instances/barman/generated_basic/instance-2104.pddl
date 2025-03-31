(define (problem prob)
 (:domain barman)
 (:objects 
      shaker11 - shaker
      left right - hand
      shot10 shot168 shot62 - shot
      ingredient114 ingredient298 - ingredient
      cocktail257 - cocktail
      dispenser419 dispenser455 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker11)
  (ontable shot10)
  (ontable shot168)
  (ontable shot62)
  (dispenses dispenser419 ingredient114)
  (dispenses dispenser455 ingredient298)
  (clean shaker11)
  (clean shot10)
  (clean shot168)
  (clean shot62)
  (empty shaker11)
  (empty shot10)
  (empty shot168)
  (empty shot62)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker11 l0)
  (shaker-level shaker11 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail257 ingredient114)
  (cocktail-part2 cocktail257 ingredient298)
)
 (:goal
  (and
      (contains shot10 cocktail257)
      (contains shot168 cocktail257)
)))
