(define (problem prob)
 (:domain barman)
 (:objects 
      shaker221 - shaker
      left right - hand
      shot377 shot102 - shot
      ingredient281 ingredient249 - ingredient
      cocktail1 - cocktail
      dispenser315 dispenser30 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker221)
  (ontable shot377)
  (ontable shot102)
  (dispenses dispenser315 ingredient281)
  (dispenses dispenser30 ingredient249)
  (clean shaker221)
  (clean shot377)
  (clean shot102)
  (empty shaker221)
  (empty shot377)
  (empty shot102)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker221 l0)
  (shaker-level shaker221 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient249)
  (cocktail-part2 cocktail1 ingredient281)
)
 (:goal
  (and
      (contains shot377 cocktail1)
)))
