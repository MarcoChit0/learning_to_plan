(define (problem prob)
 (:domain barman)
 (:objects 
      shaker45 - shaker
      left right - hand
      shot95 shot28 - shot
      ingredient162 ingredient89 - ingredient
      cocktail1 - cocktail
      dispenser499 dispenser29 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker45)
  (ontable shot95)
  (ontable shot28)
  (dispenses dispenser499 ingredient162)
  (dispenses dispenser29 ingredient89)
  (clean shaker45)
  (clean shot95)
  (clean shot28)
  (empty shaker45)
  (empty shot95)
  (empty shot28)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker45 l0)
  (shaker-level shaker45 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient162)
  (cocktail-part2 cocktail1 ingredient89)
)
 (:goal
  (and
      (contains shot95 cocktail1)
)))
