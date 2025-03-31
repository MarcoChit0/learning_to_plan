(define (problem prob)
 (:domain barman)
 (:objects 
      shaker462 - shaker
      left right - hand
      shot25 shot160 - shot
      ingredient457 ingredient472 - ingredient
      cocktail1 - cocktail
      dispenser169 dispenser97 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker462)
  (ontable shot25)
  (ontable shot160)
  (dispenses dispenser169 ingredient457)
  (dispenses dispenser97 ingredient472)
  (clean shaker462)
  (clean shot25)
  (clean shot160)
  (empty shaker462)
  (empty shot25)
  (empty shot160)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker462 l0)
  (shaker-level shaker462 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient472)
  (cocktail-part2 cocktail1 ingredient457)
)
 (:goal
  (and
      (contains shot25 cocktail1)
)))
