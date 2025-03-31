(define (problem prob)
 (:domain barman)
 (:objects 
      shaker485 - shaker
      left right - hand
      shot247 shot481 shot113 - shot
      ingredient302 ingredient203 ingredient432 ingredient38 - ingredient
      cocktail61 - cocktail
      dispenser130 dispenser41 dispenser304 dispenser316 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker485)
  (ontable shot247)
  (ontable shot481)
  (ontable shot113)
  (dispenses dispenser130 ingredient302)
  (dispenses dispenser41 ingredient203)
  (dispenses dispenser304 ingredient432)
  (dispenses dispenser316 ingredient38)
  (clean shaker485)
  (clean shot247)
  (clean shot481)
  (clean shot113)
  (empty shaker485)
  (empty shot247)
  (empty shot481)
  (empty shot113)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker485 l0)
  (shaker-level shaker485 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail61 ingredient302)
  (cocktail-part2 cocktail61 ingredient38)
)
 (:goal
  (and
      (contains shot247 cocktail61)
      (contains shot481 cocktail61)
)))
