(define (problem prob)
 (:domain barman)
 (:objects 
      shaker163 - shaker
      left right - hand
      shot61 shot428 - shot
      ingredient180 ingredient3 ingredient119 - ingredient
      cocktail190 - cocktail
      dispenser126 dispenser171 dispenser475 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker163)
  (ontable shot61)
  (ontable shot428)
  (dispenses dispenser126 ingredient180)
  (dispenses dispenser171 ingredient3)
  (dispenses dispenser475 ingredient119)
  (clean shaker163)
  (clean shot61)
  (clean shot428)
  (empty shaker163)
  (empty shot61)
  (empty shot428)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker163 l0)
  (shaker-level shaker163 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail190 ingredient119)
  (cocktail-part2 cocktail190 ingredient180)
)
 (:goal
  (and
      (contains shot61 cocktail190)
)))
