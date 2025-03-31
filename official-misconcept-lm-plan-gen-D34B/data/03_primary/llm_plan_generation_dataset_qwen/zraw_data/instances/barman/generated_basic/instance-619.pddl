(define (problem prob)
 (:domain barman)
 (:objects 
      shaker460 - shaker
      left right - hand
      shot407 shot105 - shot
      ingredient41 ingredient243 ingredient109 ingredient280 - ingredient
      cocktail1 - cocktail
      dispenser370 dispenser207 dispenser350 dispenser474 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker460)
  (ontable shot407)
  (ontable shot105)
  (dispenses dispenser370 ingredient41)
  (dispenses dispenser207 ingredient243)
  (dispenses dispenser350 ingredient109)
  (dispenses dispenser474 ingredient280)
  (clean shaker460)
  (clean shot407)
  (clean shot105)
  (empty shaker460)
  (empty shot407)
  (empty shot105)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker460 l0)
  (shaker-level shaker460 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient109)
  (cocktail-part2 cocktail1 ingredient243)
)
 (:goal
  (and
      (contains shot407 cocktail1)
)))
