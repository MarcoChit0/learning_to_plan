(define (problem prob)
 (:domain barman)
 (:objects 
      shaker41 - shaker
      left right - hand
      shot221 shot112 shot374 - shot
      ingredient90 ingredient87 - ingredient
      cocktail469 - cocktail
      dispenser87 dispenser399 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker41)
  (ontable shot221)
  (ontable shot112)
  (ontable shot374)
  (dispenses dispenser87 ingredient90)
  (dispenses dispenser399 ingredient87)
  (clean shaker41)
  (clean shot221)
  (clean shot112)
  (clean shot374)
  (empty shaker41)
  (empty shot221)
  (empty shot112)
  (empty shot374)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker41 l0)
  (shaker-level shaker41 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail469 ingredient87)
  (cocktail-part2 cocktail469 ingredient90)
)
 (:goal
  (and
      (contains shot221 cocktail469)
      (contains shot112 cocktail469)
)))
