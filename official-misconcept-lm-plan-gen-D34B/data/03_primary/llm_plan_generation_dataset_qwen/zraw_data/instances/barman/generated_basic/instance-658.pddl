(define (problem prob)
 (:domain barman)
 (:objects 
      shaker194 - shaker
      left right - hand
      shot47 shot322 - shot
      ingredient371 ingredient364 ingredient239 - ingredient
      cocktail1 - cocktail
      dispenser77 dispenser376 dispenser11 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker194)
  (ontable shot47)
  (ontable shot322)
  (dispenses dispenser77 ingredient371)
  (dispenses dispenser376 ingredient364)
  (dispenses dispenser11 ingredient239)
  (clean shaker194)
  (clean shot47)
  (clean shot322)
  (empty shaker194)
  (empty shot47)
  (empty shot322)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker194 l0)
  (shaker-level shaker194 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient371)
  (cocktail-part2 cocktail1 ingredient364)
)
 (:goal
  (and
      (contains shot47 cocktail1)
)))
