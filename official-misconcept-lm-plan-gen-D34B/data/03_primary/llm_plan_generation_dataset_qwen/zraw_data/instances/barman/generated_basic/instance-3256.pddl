(define (problem prob)
 (:domain barman)
 (:objects 
      shaker251 - shaker
      left right - hand
      shot80 shot195 shot47 - shot
      ingredient113 ingredient92 ingredient325 ingredient37 - ingredient
      cocktail322 - cocktail
      dispenser118 dispenser22 dispenser87 dispenser389 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker251)
  (ontable shot80)
  (ontable shot195)
  (ontable shot47)
  (dispenses dispenser118 ingredient113)
  (dispenses dispenser22 ingredient92)
  (dispenses dispenser87 ingredient325)
  (dispenses dispenser389 ingredient37)
  (clean shaker251)
  (clean shot80)
  (clean shot195)
  (clean shot47)
  (empty shaker251)
  (empty shot80)
  (empty shot195)
  (empty shot47)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker251 l0)
  (shaker-level shaker251 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail322 ingredient92)
  (cocktail-part2 cocktail322 ingredient325)
)
 (:goal
  (and
      (contains shot80 cocktail322)
      (contains shot195 cocktail322)
)))
