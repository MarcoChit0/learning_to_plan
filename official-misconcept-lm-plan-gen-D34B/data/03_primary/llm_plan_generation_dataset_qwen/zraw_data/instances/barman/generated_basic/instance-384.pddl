(define (problem prob)
 (:domain barman)
 (:objects 
      shaker273 - shaker
      left right - hand
      shot467 shot437 shot450 - shot
      ingredient49 ingredient118 ingredient168 - ingredient
      cocktail1 - cocktail
      dispenser272 dispenser93 dispenser339 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker273)
  (ontable shot467)
  (ontable shot437)
  (ontable shot450)
  (dispenses dispenser272 ingredient49)
  (dispenses dispenser93 ingredient118)
  (dispenses dispenser339 ingredient168)
  (clean shaker273)
  (clean shot467)
  (clean shot437)
  (clean shot450)
  (empty shaker273)
  (empty shot467)
  (empty shot437)
  (empty shot450)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker273 l0)
  (shaker-level shaker273 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient118)
  (cocktail-part2 cocktail1 ingredient49)
)
 (:goal
  (and
      (contains shot467 cocktail1)
      (contains shot437 cocktail1)
)))
