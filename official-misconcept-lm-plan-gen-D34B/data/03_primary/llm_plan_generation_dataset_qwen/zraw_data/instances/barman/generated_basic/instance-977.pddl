(define (problem prob)
 (:domain barman)
 (:objects 
      shaker409 - shaker
      left right - hand
      shot67 shot66 - shot
      ingredient201 ingredient288 ingredient365 ingredient242 - ingredient
      cocktail1 - cocktail
      dispenser214 dispenser77 dispenser110 dispenser22 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker409)
  (ontable shot67)
  (ontable shot66)
  (dispenses dispenser214 ingredient201)
  (dispenses dispenser77 ingredient288)
  (dispenses dispenser110 ingredient365)
  (dispenses dispenser22 ingredient242)
  (clean shaker409)
  (clean shot67)
  (clean shot66)
  (empty shaker409)
  (empty shot67)
  (empty shot66)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker409 l0)
  (shaker-level shaker409 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient201)
  (cocktail-part2 cocktail1 ingredient365)
)
 (:goal
  (and
      (contains shot67 cocktail1)
)))
