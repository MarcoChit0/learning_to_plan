(define (problem prob)
 (:domain barman)
 (:objects 
      shaker22 - shaker
      left right - hand
      shot355 - shot
      ingredient415 ingredient382 ingredient255 ingredient36 - ingredient
      cocktail280 - cocktail
      dispenser290 dispenser410 dispenser194 dispenser126 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker22)
  (ontable shot355)
  (dispenses dispenser290 ingredient415)
  (dispenses dispenser410 ingredient382)
  (dispenses dispenser194 ingredient255)
  (dispenses dispenser126 ingredient36)
  (clean shaker22)
  (clean shot355)
  (empty shaker22)
  (empty shot355)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker22 l0)
  (shaker-level shaker22 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail280 ingredient382)
  (cocktail-part2 cocktail280 ingredient415)
)
 (:goal
  (and
      (contains shot355 cocktail280)
)))
