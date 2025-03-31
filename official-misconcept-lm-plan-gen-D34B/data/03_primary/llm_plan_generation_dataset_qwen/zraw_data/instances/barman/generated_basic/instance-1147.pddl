(define (problem prob)
 (:domain barman)
 (:objects 
      shaker365 - shaker
      left right - hand
      shot333 shot271 - shot
      ingredient169 ingredient446 ingredient283 ingredient382 - ingredient
      cocktail1 - cocktail
      dispenser321 dispenser327 dispenser40 dispenser18 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker365)
  (ontable shot333)
  (ontable shot271)
  (dispenses dispenser321 ingredient169)
  (dispenses dispenser327 ingredient446)
  (dispenses dispenser40 ingredient283)
  (dispenses dispenser18 ingredient382)
  (clean shaker365)
  (clean shot333)
  (clean shot271)
  (empty shaker365)
  (empty shot333)
  (empty shot271)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker365 l0)
  (shaker-level shaker365 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient283)
  (cocktail-part2 cocktail1 ingredient169)
)
 (:goal
  (and
      (contains shot333 cocktail1)
)))
