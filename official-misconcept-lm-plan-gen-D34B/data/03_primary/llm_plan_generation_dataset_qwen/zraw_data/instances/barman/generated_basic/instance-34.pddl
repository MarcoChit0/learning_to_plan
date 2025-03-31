(define (problem prob)
 (:domain barman)
 (:objects 
      shaker350 - shaker
      left right - hand
      shot18 shot499 shot31 - shot
      ingredient39 ingredient15 ingredient184 ingredient329 - ingredient
      cocktail1 - cocktail
      dispenser336 dispenser469 dispenser389 dispenser31 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker350)
  (ontable shot18)
  (ontable shot499)
  (ontable shot31)
  (dispenses dispenser336 ingredient39)
  (dispenses dispenser469 ingredient15)
  (dispenses dispenser389 ingredient184)
  (dispenses dispenser31 ingredient329)
  (clean shaker350)
  (clean shot18)
  (clean shot499)
  (clean shot31)
  (empty shaker350)
  (empty shot18)
  (empty shot499)
  (empty shot31)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker350 l0)
  (shaker-level shaker350 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient184)
  (cocktail-part2 cocktail1 ingredient39)
)
 (:goal
  (and
      (contains shot18 cocktail1)
      (contains shot499 ingredient15)
)))
