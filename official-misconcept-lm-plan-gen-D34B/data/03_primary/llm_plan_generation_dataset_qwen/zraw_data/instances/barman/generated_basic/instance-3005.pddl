(define (problem prob)
 (:domain barman)
 (:objects 
      shaker407 - shaker
      left right - hand
      shot64 shot255 - shot
      ingredient424 ingredient329 ingredient298 - ingredient
      cocktail313 - cocktail
      dispenser220 dispenser471 dispenser102 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker407)
  (ontable shot64)
  (ontable shot255)
  (dispenses dispenser220 ingredient424)
  (dispenses dispenser471 ingredient329)
  (dispenses dispenser102 ingredient298)
  (clean shaker407)
  (clean shot64)
  (clean shot255)
  (empty shaker407)
  (empty shot64)
  (empty shot255)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker407 l0)
  (shaker-level shaker407 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail313 ingredient329)
  (cocktail-part2 cocktail313 ingredient298)
)
 (:goal
  (and
      (contains shot64 cocktail313)
)))
