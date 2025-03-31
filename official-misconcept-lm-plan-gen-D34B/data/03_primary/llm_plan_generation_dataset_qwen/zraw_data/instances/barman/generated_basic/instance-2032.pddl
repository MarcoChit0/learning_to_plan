(define (problem prob)
 (:domain barman)
 (:objects 
      shaker2 - shaker
      left right - hand
      shot87 shot391 - shot
      ingredient118 ingredient81 - ingredient
      cocktail43 - cocktail
      dispenser165 dispenser16 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker2)
  (ontable shot87)
  (ontable shot391)
  (dispenses dispenser165 ingredient118)
  (dispenses dispenser16 ingredient81)
  (clean shaker2)
  (clean shot87)
  (clean shot391)
  (empty shaker2)
  (empty shot87)
  (empty shot391)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker2 l0)
  (shaker-level shaker2 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail43 ingredient118)
  (cocktail-part2 cocktail43 ingredient81)
)
 (:goal
  (and
      (contains shot87 cocktail43)
)))
