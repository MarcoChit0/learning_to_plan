(define (problem prob)
 (:domain barman)
 (:objects 
      shaker493 - shaker
      left right - hand
      shot127 shot389 shot51 - shot
      ingredient321 ingredient214 - ingredient
      cocktail1 cocktail2 - cocktail
      dispenser421 dispenser260 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker493)
  (ontable shot127)
  (ontable shot389)
  (ontable shot51)
  (dispenses dispenser421 ingredient321)
  (dispenses dispenser260 ingredient214)
  (clean shaker493)
  (clean shot127)
  (clean shot389)
  (clean shot51)
  (empty shaker493)
  (empty shot127)
  (empty shot389)
  (empty shot51)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker493 l0)
  (shaker-level shaker493 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient321)
  (cocktail-part2 cocktail1 ingredient214)
  (cocktail-part1 cocktail2 ingredient214)
  (cocktail-part2 cocktail2 ingredient321)
)
 (:goal
  (and
      (contains shot127 cocktail1)
      (contains shot389 cocktail2)
)))
