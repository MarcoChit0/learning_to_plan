(define (problem prob)
 (:domain barman)
 (:objects 
      shaker41 - shaker
      left right - hand
      shot321 shot126 shot176 - shot
      ingredient8 ingredient497 - ingredient
      cocktail1 cocktail2 - cocktail
      dispenser87 dispenser468 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker41)
  (ontable shot321)
  (ontable shot126)
  (ontable shot176)
  (dispenses dispenser87 ingredient8)
  (dispenses dispenser468 ingredient497)
  (clean shaker41)
  (clean shot321)
  (clean shot126)
  (clean shot176)
  (empty shaker41)
  (empty shot321)
  (empty shot126)
  (empty shot176)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker41 l0)
  (shaker-level shaker41 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient8)
  (cocktail-part2 cocktail1 ingredient497)
  (cocktail-part1 cocktail2 ingredient497)
  (cocktail-part2 cocktail2 ingredient8)
)
 (:goal
  (and
      (contains shot321 cocktail2)
      (contains shot126 cocktail1)
)))
