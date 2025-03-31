(define (problem prob)
 (:domain barman)
 (:objects 
      shaker440 - shaker
      left right - hand
      shot296 - shot
      ingredient63 ingredient322 ingredient198 ingredient364 - ingredient
      cocktail141 - cocktail
      dispenser327 dispenser282 dispenser339 dispenser122 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker440)
  (ontable shot296)
  (dispenses dispenser327 ingredient63)
  (dispenses dispenser282 ingredient322)
  (dispenses dispenser339 ingredient198)
  (dispenses dispenser122 ingredient364)
  (clean shaker440)
  (clean shot296)
  (empty shaker440)
  (empty shot296)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker440 l0)
  (shaker-level shaker440 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail141 ingredient364)
  (cocktail-part2 cocktail141 ingredient322)
)
 (:goal
  (and
      (contains shot296 cocktail141)
)))
