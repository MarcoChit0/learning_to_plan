(define (problem prob)
 (:domain barman)
 (:objects 
      shaker17 - shaker
      left right - hand
      shot365 shot290 shot408 - shot
      ingredient211 ingredient289 ingredient425 - ingredient
      cocktail62 - cocktail
      dispenser220 dispenser19 dispenser90 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker17)
  (ontable shot365)
  (ontable shot290)
  (ontable shot408)
  (dispenses dispenser220 ingredient211)
  (dispenses dispenser19 ingredient289)
  (dispenses dispenser90 ingredient425)
  (clean shaker17)
  (clean shot365)
  (clean shot290)
  (clean shot408)
  (empty shaker17)
  (empty shot365)
  (empty shot290)
  (empty shot408)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker17 l0)
  (shaker-level shaker17 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail62 ingredient289)
  (cocktail-part2 cocktail62 ingredient425)
)
 (:goal
  (and
      (contains shot365 cocktail62)
      (contains shot290 cocktail62)
)))
