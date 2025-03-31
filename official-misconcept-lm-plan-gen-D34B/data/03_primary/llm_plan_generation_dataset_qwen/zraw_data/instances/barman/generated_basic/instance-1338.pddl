(define (problem prob)
 (:domain barman)
 (:objects 
      shaker280 - shaker
      left right - hand
      shot2 shot452 shot494 - shot
      ingredient87 ingredient336 ingredient398 - ingredient
      cocktail2 - cocktail
      dispenser0 dispenser316 dispenser89 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker280)
  (ontable shot2)
  (ontable shot452)
  (ontable shot494)
  (dispenses dispenser0 ingredient87)
  (dispenses dispenser316 ingredient336)
  (dispenses dispenser89 ingredient398)
  (clean shaker280)
  (clean shot2)
  (clean shot452)
  (clean shot494)
  (empty shaker280)
  (empty shot2)
  (empty shot452)
  (empty shot494)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker280 l0)
  (shaker-level shaker280 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail2 ingredient87)
  (cocktail-part2 cocktail2 ingredient336)
)
 (:goal
  (and
      (contains shot2 cocktail2)
      (contains shot452 cocktail2)
)))
