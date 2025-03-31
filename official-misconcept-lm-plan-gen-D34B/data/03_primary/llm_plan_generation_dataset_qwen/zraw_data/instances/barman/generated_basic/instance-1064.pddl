(define (problem prob)
 (:domain barman)
 (:objects 
      shaker297 - shaker
      left right - hand
      shot484 shot407 shot359 - shot
      ingredient270 ingredient240 ingredient14 ingredient331 - ingredient
      cocktail1 - cocktail
      dispenser80 dispenser345 dispenser205 dispenser21 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker297)
  (ontable shot484)
  (ontable shot407)
  (ontable shot359)
  (dispenses dispenser80 ingredient270)
  (dispenses dispenser345 ingredient240)
  (dispenses dispenser205 ingredient14)
  (dispenses dispenser21 ingredient331)
  (clean shaker297)
  (clean shot484)
  (clean shot407)
  (clean shot359)
  (empty shaker297)
  (empty shot484)
  (empty shot407)
  (empty shot359)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker297 l0)
  (shaker-level shaker297 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient270)
  (cocktail-part2 cocktail1 ingredient14)
)
 (:goal
  (and
      (contains shot484 cocktail1)
      (contains shot407 cocktail1)
)))
