(define (problem prob)
 (:domain barman)
 (:objects 
      shaker499 - shaker
      left right - hand
      shot353 - shot
      ingredient43 ingredient227 ingredient458 - ingredient
      cocktail1 - cocktail
      dispenser282 dispenser209 dispenser318 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker499)
  (ontable shot353)
  (dispenses dispenser282 ingredient43)
  (dispenses dispenser209 ingredient227)
  (dispenses dispenser318 ingredient458)
  (clean shaker499)
  (clean shot353)
  (empty shaker499)
  (empty shot353)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker499 l0)
  (shaker-level shaker499 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient227)
  (cocktail-part2 cocktail1 ingredient458)
)
 (:goal
  (and
      (contains shot353 cocktail1)
)))
