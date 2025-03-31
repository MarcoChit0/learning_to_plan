(define (problem prob)
 (:domain barman)
 (:objects 
      shaker436 - shaker
      left right - hand
      shot138 shot276 shot135 - shot
      ingredient186 ingredient27 ingredient69 - ingredient
      cocktail1 - cocktail
      dispenser467 dispenser176 dispenser248 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker436)
  (ontable shot138)
  (ontable shot276)
  (ontable shot135)
  (dispenses dispenser467 ingredient186)
  (dispenses dispenser176 ingredient27)
  (dispenses dispenser248 ingredient69)
  (clean shaker436)
  (clean shot138)
  (clean shot276)
  (clean shot135)
  (empty shaker436)
  (empty shot138)
  (empty shot276)
  (empty shot135)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker436 l0)
  (shaker-level shaker436 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient27)
  (cocktail-part2 cocktail1 ingredient186)
)
 (:goal
  (and
      (contains shot138 cocktail1)
      (contains shot276 cocktail1)
)))
