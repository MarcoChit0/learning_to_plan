(define (problem prob)
 (:domain barman)
 (:objects 
      shaker453 - shaker
      left right - hand
      shot122 shot143 shot469 - shot
      ingredient171 ingredient299 ingredient255 - ingredient
      cocktail460 - cocktail
      dispenser423 dispenser479 dispenser490 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker453)
  (ontable shot122)
  (ontable shot143)
  (ontable shot469)
  (dispenses dispenser423 ingredient171)
  (dispenses dispenser479 ingredient299)
  (dispenses dispenser490 ingredient255)
  (clean shaker453)
  (clean shot122)
  (clean shot143)
  (clean shot469)
  (empty shaker453)
  (empty shot122)
  (empty shot143)
  (empty shot469)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker453 l0)
  (shaker-level shaker453 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail460 ingredient299)
  (cocktail-part2 cocktail460 ingredient171)
)
 (:goal
  (and
      (contains shot122 cocktail460)
      (contains shot143 ingredient255)
)))
