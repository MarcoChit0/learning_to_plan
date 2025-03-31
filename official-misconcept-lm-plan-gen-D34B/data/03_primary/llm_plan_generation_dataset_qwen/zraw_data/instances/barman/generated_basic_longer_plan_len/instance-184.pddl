(define (problem prob)
 (:domain barman)
 (:objects 
      shaker459 - shaker
      left right - hand
      shot378 shot194 shot105 - shot
      ingredient60 ingredient344 - ingredient
      cocktail1 cocktail2 - cocktail
      dispenser157 dispenser97 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker459)
  (ontable shot378)
  (ontable shot194)
  (ontable shot105)
  (dispenses dispenser157 ingredient60)
  (dispenses dispenser97 ingredient344)
  (clean shaker459)
  (clean shot378)
  (clean shot194)
  (clean shot105)
  (empty shaker459)
  (empty shot378)
  (empty shot194)
  (empty shot105)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker459 l0)
  (shaker-level shaker459 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient60)
  (cocktail-part2 cocktail1 ingredient344)
  (cocktail-part1 cocktail2 ingredient344)
  (cocktail-part2 cocktail2 ingredient60)
)
 (:goal
  (and
      (contains shot378 cocktail1)
      (contains shot194 cocktail2)
)))
