(define (problem prob)
 (:domain barman)
 (:objects 
      shaker152 - shaker
      left right - hand
      shot359 shot100 shot25 - shot
      ingredient174 ingredient243 ingredient255 - ingredient
      cocktail1 - cocktail
      dispenser234 dispenser206 dispenser410 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker152)
  (ontable shot359)
  (ontable shot100)
  (ontable shot25)
  (dispenses dispenser234 ingredient174)
  (dispenses dispenser206 ingredient243)
  (dispenses dispenser410 ingredient255)
  (clean shaker152)
  (clean shot359)
  (clean shot100)
  (clean shot25)
  (empty shaker152)
  (empty shot359)
  (empty shot100)
  (empty shot25)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker152 l0)
  (shaker-level shaker152 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient255)
  (cocktail-part2 cocktail1 ingredient243)
)
 (:goal
  (and
      (contains shot359 cocktail1)
      (contains shot100 ingredient255)
)))
