(define (problem prob)
 (:domain barman)
 (:objects 
      shaker432 - shaker
      left right - hand
      shot56 shot206 - shot
      ingredient242 ingredient409 ingredient16 - ingredient
      cocktail1 - cocktail
      dispenser316 dispenser481 dispenser300 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker432)
  (ontable shot56)
  (ontable shot206)
  (dispenses dispenser316 ingredient242)
  (dispenses dispenser481 ingredient409)
  (dispenses dispenser300 ingredient16)
  (clean shaker432)
  (clean shot56)
  (clean shot206)
  (empty shaker432)
  (empty shot56)
  (empty shot206)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker432 l0)
  (shaker-level shaker432 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient242)
  (cocktail-part2 cocktail1 ingredient16)
)
 (:goal
  (and
      (contains shot56 cocktail1)
)))
