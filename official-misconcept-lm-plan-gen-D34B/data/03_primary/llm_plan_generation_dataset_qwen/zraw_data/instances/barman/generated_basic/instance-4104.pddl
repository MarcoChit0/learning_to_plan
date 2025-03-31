(define (problem prob)
 (:domain barman)
 (:objects 
      shaker188 - shaker
      left right - hand
      shot299 shot391 - shot
      ingredient14 ingredient168 - ingredient
      cocktail399 - cocktail
      dispenser127 dispenser26 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker188)
  (ontable shot299)
  (ontable shot391)
  (dispenses dispenser127 ingredient14)
  (dispenses dispenser26 ingredient168)
  (clean shaker188)
  (clean shot299)
  (clean shot391)
  (empty shaker188)
  (empty shot299)
  (empty shot391)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker188 l0)
  (shaker-level shaker188 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail399 ingredient168)
  (cocktail-part2 cocktail399 ingredient14)
)
 (:goal
  (and
      (contains shot299 cocktail399)
)))
