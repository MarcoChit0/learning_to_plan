(define (problem prob)
 (:domain barman)
 (:objects 
      shaker453 - shaker
      left right - hand
      shot413 - shot
      ingredient314 ingredient421 ingredient321 - ingredient
      cocktail1 - cocktail
      dispenser141 dispenser478 dispenser271 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker453)
  (ontable shot413)
  (dispenses dispenser141 ingredient314)
  (dispenses dispenser478 ingredient421)
  (dispenses dispenser271 ingredient321)
  (clean shaker453)
  (clean shot413)
  (empty shaker453)
  (empty shot413)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker453 l0)
  (shaker-level shaker453 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient421)
  (cocktail-part2 cocktail1 ingredient314)
)
 (:goal
  (and
      (contains shot413 cocktail1)
)))
