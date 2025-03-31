(define (problem prob)
 (:domain barman)
 (:objects 
      shaker110 - shaker
      left right - hand
      shot187 shot329 - shot
      ingredient327 ingredient82 ingredient209 - ingredient
      cocktail460 - cocktail
      dispenser475 dispenser320 dispenser194 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker110)
  (ontable shot187)
  (ontable shot329)
  (dispenses dispenser475 ingredient327)
  (dispenses dispenser320 ingredient82)
  (dispenses dispenser194 ingredient209)
  (clean shaker110)
  (clean shot187)
  (clean shot329)
  (empty shaker110)
  (empty shot187)
  (empty shot329)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker110 l0)
  (shaker-level shaker110 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail460 ingredient209)
  (cocktail-part2 cocktail460 ingredient82)
)
 (:goal
  (and
      (contains shot187 cocktail460)
)))
