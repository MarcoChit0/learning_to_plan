(define (problem prob)
 (:domain barman)
 (:objects 
      shaker82 - shaker
      left right - hand
      shot260 shot377 - shot
      ingredient35 ingredient39 - ingredient
      cocktail111 - cocktail
      dispenser491 dispenser29 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker82)
  (ontable shot260)
  (ontable shot377)
  (dispenses dispenser491 ingredient35)
  (dispenses dispenser29 ingredient39)
  (clean shaker82)
  (clean shot260)
  (clean shot377)
  (empty shaker82)
  (empty shot260)
  (empty shot377)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker82 l0)
  (shaker-level shaker82 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail111 ingredient39)
  (cocktail-part2 cocktail111 ingredient35)
)
 (:goal
  (and
      (contains shot260 cocktail111)
)))
