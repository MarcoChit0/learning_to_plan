(define (problem prob)
 (:domain barman)
 (:objects 
      shaker481 - shaker
      left right - hand
      shot228 shot353 shot468 - shot
      ingredient288 ingredient472 - ingredient
      cocktail223 - cocktail
      dispenser431 dispenser344 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker481)
  (ontable shot228)
  (ontable shot353)
  (ontable shot468)
  (dispenses dispenser431 ingredient288)
  (dispenses dispenser344 ingredient472)
  (clean shaker481)
  (clean shot228)
  (clean shot353)
  (clean shot468)
  (empty shaker481)
  (empty shot228)
  (empty shot353)
  (empty shot468)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker481 l0)
  (shaker-level shaker481 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail223 ingredient288)
  (cocktail-part2 cocktail223 ingredient472)
)
 (:goal
  (and
      (contains shot228 cocktail223)
      (contains shot353 cocktail223)
)))
