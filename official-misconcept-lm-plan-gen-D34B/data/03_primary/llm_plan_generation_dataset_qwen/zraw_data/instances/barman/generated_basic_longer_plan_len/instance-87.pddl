(define (problem prob)
 (:domain barman)
 (:objects 
      shaker484 - shaker
      left right - hand
      shot97 shot118 shot106 - shot
      ingredient332 ingredient218 ingredient286 - ingredient
      cocktail1 cocktail2 - cocktail
      dispenser495 dispenser125 dispenser156 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker484)
  (ontable shot97)
  (ontable shot118)
  (ontable shot106)
  (dispenses dispenser495 ingredient332)
  (dispenses dispenser125 ingredient218)
  (dispenses dispenser156 ingredient286)
  (clean shaker484)
  (clean shot97)
  (clean shot118)
  (clean shot106)
  (empty shaker484)
  (empty shot97)
  (empty shot118)
  (empty shot106)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker484 l0)
  (shaker-level shaker484 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient286)
  (cocktail-part2 cocktail1 ingredient218)
  (cocktail-part1 cocktail2 ingredient332)
  (cocktail-part2 cocktail2 ingredient286)
)
 (:goal
  (and
      (contains shot97 cocktail1)
      (contains shot118 cocktail2)
)))
