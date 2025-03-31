(define (problem prob)
 (:domain barman)
 (:objects 
      shaker16 - shaker
      left right - hand
      shot481 shot48 shot181 - shot
      ingredient401 ingredient493 ingredient148 - ingredient
      cocktail1 - cocktail
      dispenser445 dispenser16 dispenser307 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker16)
  (ontable shot481)
  (ontable shot48)
  (ontable shot181)
  (dispenses dispenser445 ingredient401)
  (dispenses dispenser16 ingredient493)
  (dispenses dispenser307 ingredient148)
  (clean shaker16)
  (clean shot481)
  (clean shot48)
  (clean shot181)
  (empty shaker16)
  (empty shot481)
  (empty shot48)
  (empty shot181)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker16 l0)
  (shaker-level shaker16 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient493)
  (cocktail-part2 cocktail1 ingredient148)
)
 (:goal
  (and
      (contains shot481 cocktail1)
      (contains shot48 ingredient493)
)))
