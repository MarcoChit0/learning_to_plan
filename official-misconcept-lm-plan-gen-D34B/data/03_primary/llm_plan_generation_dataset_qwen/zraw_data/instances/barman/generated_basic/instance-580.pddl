(define (problem prob)
 (:domain barman)
 (:objects 
      shaker187 - shaker
      left right - hand
      shot47 shot75 - shot
      ingredient55 ingredient494 - ingredient
      cocktail1 - cocktail
      dispenser495 dispenser216 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker187)
  (ontable shot47)
  (ontable shot75)
  (dispenses dispenser495 ingredient55)
  (dispenses dispenser216 ingredient494)
  (clean shaker187)
  (clean shot47)
  (clean shot75)
  (empty shaker187)
  (empty shot47)
  (empty shot75)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker187 l0)
  (shaker-level shaker187 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient55)
  (cocktail-part2 cocktail1 ingredient494)
)
 (:goal
  (and
      (contains shot47 cocktail1)
)))
