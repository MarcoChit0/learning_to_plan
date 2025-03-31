(define (problem prob)
 (:domain barman)
 (:objects 
      shaker360 - shaker
      left right - hand
      shot3 shot434 shot203 - shot
      ingredient274 ingredient419 ingredient374 - ingredient
      cocktail334 - cocktail
      dispenser254 dispenser493 dispenser329 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker360)
  (ontable shot3)
  (ontable shot434)
  (ontable shot203)
  (dispenses dispenser254 ingredient274)
  (dispenses dispenser493 ingredient419)
  (dispenses dispenser329 ingredient374)
  (clean shaker360)
  (clean shot3)
  (clean shot434)
  (clean shot203)
  (empty shaker360)
  (empty shot3)
  (empty shot434)
  (empty shot203)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker360 l0)
  (shaker-level shaker360 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail334 ingredient419)
  (cocktail-part2 cocktail334 ingredient374)
)
 (:goal
  (and
      (contains shot3 cocktail334)
      (contains shot434 cocktail334)
)))
