(define (problem prob)
 (:domain barman)
 (:objects 
      shaker462 - shaker
      left right - hand
      shot341 shot19 - shot
      ingredient141 ingredient457 ingredient277 ingredient430 - ingredient
      cocktail1 - cocktail
      dispenser330 dispenser386 dispenser467 dispenser394 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker462)
  (ontable shot341)
  (ontable shot19)
  (dispenses dispenser330 ingredient141)
  (dispenses dispenser386 ingredient457)
  (dispenses dispenser467 ingredient277)
  (dispenses dispenser394 ingredient430)
  (clean shaker462)
  (clean shot341)
  (clean shot19)
  (empty shaker462)
  (empty shot341)
  (empty shot19)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker462 l0)
  (shaker-level shaker462 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient457)
  (cocktail-part2 cocktail1 ingredient430)
)
 (:goal
  (and
      (contains shot341 cocktail1)
)))
