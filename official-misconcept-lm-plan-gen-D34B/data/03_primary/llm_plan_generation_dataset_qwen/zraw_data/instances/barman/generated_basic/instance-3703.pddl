(define (problem prob)
 (:domain barman)
 (:objects 
      shaker373 - shaker
      left right - hand
      shot377 shot75 - shot
      ingredient285 ingredient147 ingredient460 - ingredient
      cocktail185 - cocktail
      dispenser496 dispenser339 dispenser204 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker373)
  (ontable shot377)
  (ontable shot75)
  (dispenses dispenser496 ingredient285)
  (dispenses dispenser339 ingredient147)
  (dispenses dispenser204 ingredient460)
  (clean shaker373)
  (clean shot377)
  (clean shot75)
  (empty shaker373)
  (empty shot377)
  (empty shot75)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker373 l0)
  (shaker-level shaker373 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail185 ingredient147)
  (cocktail-part2 cocktail185 ingredient285)
)
 (:goal
  (and
      (contains shot377 cocktail185)
)))
