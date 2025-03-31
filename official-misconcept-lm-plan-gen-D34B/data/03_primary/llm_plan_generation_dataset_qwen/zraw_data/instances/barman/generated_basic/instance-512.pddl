(define (problem prob)
 (:domain barman)
 (:objects 
      shaker444 - shaker
      left right - hand
      shot353 shot311 - shot
      ingredient133 ingredient436 ingredient367 ingredient100 - ingredient
      cocktail1 - cocktail
      dispenser448 dispenser439 dispenser167 dispenser188 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker444)
  (ontable shot353)
  (ontable shot311)
  (dispenses dispenser448 ingredient133)
  (dispenses dispenser439 ingredient436)
  (dispenses dispenser167 ingredient367)
  (dispenses dispenser188 ingredient100)
  (clean shaker444)
  (clean shot353)
  (clean shot311)
  (empty shaker444)
  (empty shot353)
  (empty shot311)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker444 l0)
  (shaker-level shaker444 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient100)
  (cocktail-part2 cocktail1 ingredient436)
)
 (:goal
  (and
      (contains shot353 cocktail1)
)))
