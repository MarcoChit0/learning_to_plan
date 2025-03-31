(define (problem prob)
 (:domain barman)
 (:objects 
      shaker18 - shaker
      left right - hand
      shot73 shot104 shot322 - shot
      ingredient359 ingredient112 ingredient28 - ingredient
      cocktail197 - cocktail
      dispenser163 dispenser349 dispenser303 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker18)
  (ontable shot73)
  (ontable shot104)
  (ontable shot322)
  (dispenses dispenser163 ingredient359)
  (dispenses dispenser349 ingredient112)
  (dispenses dispenser303 ingredient28)
  (clean shaker18)
  (clean shot73)
  (clean shot104)
  (clean shot322)
  (empty shaker18)
  (empty shot73)
  (empty shot104)
  (empty shot322)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker18 l0)
  (shaker-level shaker18 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail197 ingredient28)
  (cocktail-part2 cocktail197 ingredient112)
)
 (:goal
  (and
      (contains shot73 cocktail197)
      (contains shot104 cocktail197)
)))
