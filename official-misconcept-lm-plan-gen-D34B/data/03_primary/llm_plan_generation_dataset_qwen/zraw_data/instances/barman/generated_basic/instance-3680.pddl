(define (problem prob)
 (:domain barman)
 (:objects 
      shaker311 - shaker
      left right - hand
      shot498 shot49 shot438 - shot
      ingredient335 ingredient240 ingredient385 ingredient194 - ingredient
      cocktail380 - cocktail
      dispenser104 dispenser52 dispenser441 dispenser352 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker311)
  (ontable shot498)
  (ontable shot49)
  (ontable shot438)
  (dispenses dispenser104 ingredient335)
  (dispenses dispenser52 ingredient240)
  (dispenses dispenser441 ingredient385)
  (dispenses dispenser352 ingredient194)
  (clean shaker311)
  (clean shot498)
  (clean shot49)
  (clean shot438)
  (empty shaker311)
  (empty shot498)
  (empty shot49)
  (empty shot438)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker311 l0)
  (shaker-level shaker311 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail380 ingredient194)
  (cocktail-part2 cocktail380 ingredient240)
)
 (:goal
  (and
      (contains shot498 cocktail380)
      (contains shot49 ingredient240)
)))
