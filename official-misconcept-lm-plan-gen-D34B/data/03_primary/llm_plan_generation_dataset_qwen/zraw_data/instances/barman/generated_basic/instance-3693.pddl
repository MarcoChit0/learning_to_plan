(define (problem prob)
 (:domain barman)
 (:objects 
      shaker469 - shaker
      left right - hand
      shot410 shot473 - shot
      ingredient217 ingredient153 ingredient428 - ingredient
      cocktail350 - cocktail
      dispenser479 dispenser214 dispenser82 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker469)
  (ontable shot410)
  (ontable shot473)
  (dispenses dispenser479 ingredient217)
  (dispenses dispenser214 ingredient153)
  (dispenses dispenser82 ingredient428)
  (clean shaker469)
  (clean shot410)
  (clean shot473)
  (empty shaker469)
  (empty shot410)
  (empty shot473)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker469 l0)
  (shaker-level shaker469 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail350 ingredient428)
  (cocktail-part2 cocktail350 ingredient217)
)
 (:goal
  (and
      (contains shot410 cocktail350)
)))
