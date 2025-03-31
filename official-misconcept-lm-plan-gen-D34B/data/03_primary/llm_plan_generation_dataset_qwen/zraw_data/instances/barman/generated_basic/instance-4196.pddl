(define (problem prob)
 (:domain barman)
 (:objects 
      shaker116 - shaker
      left right - hand
      shot422 shot317 shot42 - shot
      ingredient315 ingredient148 ingredient427 - ingredient
      cocktail211 - cocktail
      dispenser356 dispenser323 dispenser30 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker116)
  (ontable shot422)
  (ontable shot317)
  (ontable shot42)
  (dispenses dispenser356 ingredient315)
  (dispenses dispenser323 ingredient148)
  (dispenses dispenser30 ingredient427)
  (clean shaker116)
  (clean shot422)
  (clean shot317)
  (clean shot42)
  (empty shaker116)
  (empty shot422)
  (empty shot317)
  (empty shot42)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker116 l0)
  (shaker-level shaker116 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail211 ingredient427)
  (cocktail-part2 cocktail211 ingredient148)
)
 (:goal
  (and
      (contains shot422 cocktail211)
      (contains shot317 ingredient315)
)))
