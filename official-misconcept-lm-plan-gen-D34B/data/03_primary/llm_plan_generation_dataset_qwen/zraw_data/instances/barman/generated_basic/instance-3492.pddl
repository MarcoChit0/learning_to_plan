(define (problem prob)
 (:domain barman)
 (:objects 
      shaker442 - shaker
      left right - hand
      shot251 shot333 shot35 - shot
      ingredient469 ingredient452 ingredient19 - ingredient
      cocktail433 - cocktail
      dispenser274 dispenser273 dispenser142 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker442)
  (ontable shot251)
  (ontable shot333)
  (ontable shot35)
  (dispenses dispenser274 ingredient469)
  (dispenses dispenser273 ingredient452)
  (dispenses dispenser142 ingredient19)
  (clean shaker442)
  (clean shot251)
  (clean shot333)
  (clean shot35)
  (empty shaker442)
  (empty shot251)
  (empty shot333)
  (empty shot35)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker442 l0)
  (shaker-level shaker442 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail433 ingredient452)
  (cocktail-part2 cocktail433 ingredient469)
)
 (:goal
  (and
      (contains shot251 cocktail433)
      (contains shot333 cocktail433)
)))
