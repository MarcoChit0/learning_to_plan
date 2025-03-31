(define (problem prob)
 (:domain barman)
 (:objects 
      shaker484 - shaker
      left right - hand
      shot158 shot157 - shot
      ingredient35 ingredient239 ingredient259 ingredient41 - ingredient
      cocktail1 - cocktail
      dispenser112 dispenser216 dispenser38 dispenser264 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker484)
  (ontable shot158)
  (ontable shot157)
  (dispenses dispenser112 ingredient35)
  (dispenses dispenser216 ingredient239)
  (dispenses dispenser38 ingredient259)
  (dispenses dispenser264 ingredient41)
  (clean shaker484)
  (clean shot158)
  (clean shot157)
  (empty shaker484)
  (empty shot158)
  (empty shot157)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker484 l0)
  (shaker-level shaker484 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient239)
  (cocktail-part2 cocktail1 ingredient259)
)
 (:goal
  (and
      (contains shot158 cocktail1)
)))
