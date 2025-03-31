(define (problem prob)
 (:domain barman)
 (:objects 
      shaker431 - shaker
      left right - hand
      shot172 - shot
      ingredient281 ingredient164 ingredient313 - ingredient
      cocktail453 - cocktail
      dispenser49 dispenser115 dispenser496 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker431)
  (ontable shot172)
  (dispenses dispenser49 ingredient281)
  (dispenses dispenser115 ingredient164)
  (dispenses dispenser496 ingredient313)
  (clean shaker431)
  (clean shot172)
  (empty shaker431)
  (empty shot172)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker431 l0)
  (shaker-level shaker431 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail453 ingredient164)
  (cocktail-part2 cocktail453 ingredient281)
)
 (:goal
  (and
      (contains shot172 cocktail453)
)))
