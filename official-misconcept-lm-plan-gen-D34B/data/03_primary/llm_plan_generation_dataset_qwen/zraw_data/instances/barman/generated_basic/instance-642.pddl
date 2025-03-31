(define (problem prob)
 (:domain barman)
 (:objects 
      shaker472 - shaker
      left right - hand
      shot368 shot30 shot113 - shot
      ingredient430 ingredient299 ingredient271 - ingredient
      cocktail1 - cocktail
      dispenser243 dispenser207 dispenser14 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker472)
  (ontable shot368)
  (ontable shot30)
  (ontable shot113)
  (dispenses dispenser243 ingredient430)
  (dispenses dispenser207 ingredient299)
  (dispenses dispenser14 ingredient271)
  (clean shaker472)
  (clean shot368)
  (clean shot30)
  (clean shot113)
  (empty shaker472)
  (empty shot368)
  (empty shot30)
  (empty shot113)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker472 l0)
  (shaker-level shaker472 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient430)
  (cocktail-part2 cocktail1 ingredient271)
)
 (:goal
  (and
      (contains shot368 cocktail1)
      (contains shot30 cocktail1)
)))
