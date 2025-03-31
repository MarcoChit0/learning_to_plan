(define (problem prob)
 (:domain barman)
 (:objects 
      shaker40 - shaker
      left right - hand
      shot84 shot61 - shot
      ingredient25 ingredient254 ingredient364 ingredient354 - ingredient
      cocktail390 - cocktail
      dispenser239 dispenser21 dispenser326 dispenser149 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker40)
  (ontable shot84)
  (ontable shot61)
  (dispenses dispenser239 ingredient25)
  (dispenses dispenser21 ingredient254)
  (dispenses dispenser326 ingredient364)
  (dispenses dispenser149 ingredient354)
  (clean shaker40)
  (clean shot84)
  (clean shot61)
  (empty shaker40)
  (empty shot84)
  (empty shot61)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker40 l0)
  (shaker-level shaker40 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail390 ingredient254)
  (cocktail-part2 cocktail390 ingredient25)
)
 (:goal
  (and
      (contains shot84 cocktail390)
)))
