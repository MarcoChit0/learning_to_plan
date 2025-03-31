(define (problem prob)
 (:domain barman)
 (:objects 
      shaker441 - shaker
      left right - hand
      shot468 shot416 shot293 - shot
      ingredient413 ingredient70 - ingredient
      cocktail338 - cocktail
      dispenser175 dispenser227 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker441)
  (ontable shot468)
  (ontable shot416)
  (ontable shot293)
  (dispenses dispenser175 ingredient413)
  (dispenses dispenser227 ingredient70)
  (clean shaker441)
  (clean shot468)
  (clean shot416)
  (clean shot293)
  (empty shaker441)
  (empty shot468)
  (empty shot416)
  (empty shot293)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker441 l0)
  (shaker-level shaker441 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail338 ingredient413)
  (cocktail-part2 cocktail338 ingredient70)
)
 (:goal
  (and
      (contains shot468 cocktail338)
      (contains shot416 cocktail338)
)))
