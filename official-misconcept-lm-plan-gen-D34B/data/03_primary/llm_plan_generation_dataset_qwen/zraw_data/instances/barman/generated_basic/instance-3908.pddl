(define (problem prob)
 (:domain barman)
 (:objects 
      shaker149 - shaker
      left right - hand
      shot393 - shot
      ingredient347 ingredient432 ingredient201 - ingredient
      cocktail74 - cocktail
      dispenser331 dispenser0 dispenser311 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker149)
  (ontable shot393)
  (dispenses dispenser331 ingredient347)
  (dispenses dispenser0 ingredient432)
  (dispenses dispenser311 ingredient201)
  (clean shaker149)
  (clean shot393)
  (empty shaker149)
  (empty shot393)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker149 l0)
  (shaker-level shaker149 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail74 ingredient347)
  (cocktail-part2 cocktail74 ingredient432)
)
 (:goal
  (and
      (contains shot393 cocktail74)
)))
