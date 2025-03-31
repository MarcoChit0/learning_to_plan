(define (problem prob)
 (:domain barman)
 (:objects 
      shaker322 - shaker
      left right - hand
      shot407 shot172 - shot
      ingredient432 ingredient187 ingredient342 ingredient289 - ingredient
      cocktail1 - cocktail
      dispenser440 dispenser430 dispenser33 dispenser418 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker322)
  (ontable shot407)
  (ontable shot172)
  (dispenses dispenser440 ingredient432)
  (dispenses dispenser430 ingredient187)
  (dispenses dispenser33 ingredient342)
  (dispenses dispenser418 ingredient289)
  (clean shaker322)
  (clean shot407)
  (clean shot172)
  (empty shaker322)
  (empty shot407)
  (empty shot172)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker322 l0)
  (shaker-level shaker322 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient342)
  (cocktail-part2 cocktail1 ingredient432)
)
 (:goal
  (and
      (contains shot407 cocktail1)
)))
