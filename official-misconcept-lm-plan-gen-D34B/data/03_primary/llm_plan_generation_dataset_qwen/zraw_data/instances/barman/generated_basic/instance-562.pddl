(define (problem prob)
 (:domain barman)
 (:objects 
      shaker407 - shaker
      left right - hand
      shot482 shot392 - shot
      ingredient180 ingredient454 ingredient273 - ingredient
      cocktail1 - cocktail
      dispenser16 dispenser305 dispenser274 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker407)
  (ontable shot482)
  (ontable shot392)
  (dispenses dispenser16 ingredient180)
  (dispenses dispenser305 ingredient454)
  (dispenses dispenser274 ingredient273)
  (clean shaker407)
  (clean shot482)
  (clean shot392)
  (empty shaker407)
  (empty shot482)
  (empty shot392)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker407 l0)
  (shaker-level shaker407 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient454)
  (cocktail-part2 cocktail1 ingredient180)
)
 (:goal
  (and
      (contains shot482 cocktail1)
)))
