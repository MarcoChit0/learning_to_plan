(define (problem prob)
 (:domain barman)
 (:objects 
      shaker215 - shaker
      left right - hand
      shot187 - shot
      ingredient63 ingredient19 ingredient268 - ingredient
      cocktail1 - cocktail
      dispenser446 dispenser126 dispenser417 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker215)
  (ontable shot187)
  (dispenses dispenser446 ingredient63)
  (dispenses dispenser126 ingredient19)
  (dispenses dispenser417 ingredient268)
  (clean shaker215)
  (clean shot187)
  (empty shaker215)
  (empty shot187)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker215 l0)
  (shaker-level shaker215 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient19)
  (cocktail-part2 cocktail1 ingredient268)
)
 (:goal
  (and
      (contains shot187 cocktail1)
)))
